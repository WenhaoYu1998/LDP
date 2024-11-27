# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import tempdir
from turtle import pen
import numpy as np
import torch
import argparse
import os
import gym
import time
import json
import dmc2gym
from shutil import copyfile


import utils
from logger import Logger
from video import VideoRecorder

from envs import make_env, read_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    parser.add_argument('--arrive_arr_len', default=1000, type=int)
    parser.add_argument('--SNR', default=0.9, type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='ambs', type=str, choices=['ambs'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)

    args = parser.parse_args()
    #print(args)
    return args

def add_noise(obs, args, noise_type = 'gaussian'):
    if noise_type == 'salt_pepper':
        SNR = args.SNR
        mask = np.random.choice((0, 1, 2), size=(obs.shape[0], obs.shape[1], obs.shape[2]), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        mask = np.repeat(mask, obs.shape[0], axis=0)
        obs[mask == 1] = 255
        obs[mask == 2] = 0

    if noise_type == 'gaussian':
        obs = obs / 255
        noise = np.random.normal(0, 0.1, obs.shape)
        gaussian_out = obs + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        gaussian_out = np.float32(gaussian_out * 255)
        obs = gaussian_out
    
    return obs

def evaluate(env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None, noise = True):
    # carla metrics:
    args = parse_args()
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0

    # embedding visualization
    obses = []
    values = []
    embeddings = []

    for i in range(num_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.

        obs = env.reset()
        vector_state = np.array(obs[1][0])
        obs = np.array(obs[0][0])
        if noise:
            obs = add_noise(obs, args)
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs, vector_state)

            #尝试解决最优动作边界问题
            action_env = action
            action_env = action_env * 1.1
            action_env = np.clip(action_env, -1, 1)
            #将jia速度映射到[-2,2],zhuan角映射到[-0.785，0.785]
            action_env[0] = 0.6 * (action_env[1] + 1.0) - 0.6
            action_env[1] = 0.785 * (action_env[1] + 1.0) - 0.785
            action_env = [tuple(action_env)]

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            #obs, reward, done, info = env.step(action)
            temp_state = env.step(action_env)
            obs = np.array(temp_state[0][0][0])
            if noise:
                obs = add_noise(obs, args)
            reward = temp_state[1][0]
            done = bool(temp_state[2][0])
            info = temp_state[3]
            is_arrive = bool(temp_state[3]['arrive'])

            # metrics:
            if do_carla_metrics:
                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                steer += abs(info['steer'])
                brake += info['brake']
                count += 1

            video.record(env)
            episode_reward += reward

        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
        L.dump(step)
        return is_arrive
        
    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    # L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')


def make_agent(obs_shape, vector_state_shape, action_shape, args, device):
    if args.agent == 'baseline':
        agent = BaselineAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'bisim':
        agent = BisimAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef
        )
    elif args.agent == 'deepmdp':
        agent = DeepMDPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'ambs':
        from agent.ambs_agent import AMBSRatioAgent 
        import agent.ambs_agent
        source_code_file_path = agent.ambs_agent.__file__
        agent = AMBSRatioAgent(
            obs_shape=obs_shape,
            vector_state_shape=vector_state_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
        )
        agent.source_code_file_path = source_code_file_path

    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    if args.domain_name == 'carla':
        env = CarlaEnv(
            render_display=args.render,  # for local debugging only
            display_text=args.render,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=args.image_size,
            max_episode_steps=1000,
            frame_skip=args.action_repeat,
            is_other_cars=True,
            port=args.port
        )
        # TODO: implement env.seed(args.seed) ?

        eval_env = env
    elif args.domain_name == 'drl_env':
        cfg = read_yaml('envs/cfg/circle.yaml')
        env = make_env(cfg)
        env.seed(args.seed)
        eval_env = env
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.eval_resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )

    if args.domain_name == 'drl_env':
    # stack several consecutive frames together
        if args.encoder_type.startswith('pixel'):
            env = utils.FrameStack(env, cfg, k=args.frame_stack)
            eval_env = utils.FrameStack(eval_env, cfg, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    replay_buffer = utils.ReplayBuffer(
        #obs_shape=env.observation_space.shape,
        obs_shape=(args.frame_stack, cfg['image_size'][0], cfg['image_size'][1]),
        vector_state_shape=(args.frame_stack, cfg['state_dim'] * cfg['state_batch']),
        #action_shape=env.action_space.shape,
        action_shape=(2,),
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        #is_framestack=True
    )

    agent = make_agent(
        #obs_shape=env.observation_space.shape,
        obs_shape=(cfg['image_batch'], cfg['image_size'][0], cfg['image_size'][1]),
        vector_state_shape=(cfg['state_dim'] * cfg['state_batch'],),
        #action_shape=env.action_space.shape,
        action_shape=(2,),
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if agent.source_code_file_path is not None:
        code_dir = os.path.join(args.work_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)
        copyfile(agent.source_code_file_path, os.path.join(code_dir, 
                os.path.basename(agent.source_code_file_path)))

    episode = 0
    episode_reward, done, is_arrive = [], [], []
    arrive_arr, eval_arrive_arr = [], []
    obs, vector_state, next_obs, next_vector_state= [], [], [], []
    for i in range(cfg['agent_num_per_env']):
        episode_reward.append(0.0)
        done.append(True)
        is_arrive.append(False)
        arrive_arr.append(np.zeros([args.arrive_arr_len, 1]))
        obs.append(0.0)
        vector_state.append(0.0)
        next_obs.append(0.0)
        next_vector_state.append(0.0)
    
    agent.load('./model', 1549164)
    start_time = time.time()
    global_done = True
    for step in range(args.num_train_steps):
        if global_done:
            
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                if args.save_model and step > 1000000 and episode % 1000 == 0:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)
            
            for i in range(cfg['agent_num_per_env']):
                if is_arrive[i]:
                    arrive_arr[i][episode % args.arrive_arr_len] = 1
                else:
                    arrive_arr[i][episode % args.arrive_arr_len] = 0
                
                L.log('train/episode_reward_robot_' + str(i), episode_reward[i], step)
            
            done, episode_reward, reward = [], [], []
            observation = env.reset()
            for i in range(cfg['agent_num_per_env']):
                obs[i] = np.array(observation[0][i])
                vector_state[i] = np.array(observation[1][i])
                done.append(False)
                episode_reward.append(0.0)
                reward.append(0.0)
                arrive_rate = np.sum(arrive_arr[i]) / args.arrive_arr_len
                print('arrive_rate_robot_' + str(i) + ': {}'.format(arrive_rate))
                L.log('train/arrive_rate_robot_' + str(i), arrive_rate, step)

            episode += 1
            L.log('train/episode', episode, step)
            if episode % 100 == 0:
                env.seed(int(time.time()) % 100)

        # sample action for data collection
        #if step < args.init_steps:
        #    action = env.action_space.sample()
        #else:
        action, action_env = [], []
        for i in range(cfg['agent_num_per_env']):
            with utils.eval_mode(agent):
                action.append(agent.sample_action(obs[i], vector_state[i]))
            action_env.append(action[i] * 1.0)
            action_env[i] = np.clip(action_env[i], -1, 1)
            action_env[i][0] = 0.6 * (action_env[i][0] + 1.0) - 0.6
            action_env[i][1] = 0.785 * (action_env[i][1] + 1.0) - 0.785
            action_env[i] = tuple(action_env[i])
        # #action[0] = (action[0] + 1.0) * 0.3
        # #尝试解决最优动作边界问题
        # action_env = action
        # action_env = action_env * 1.1
        # action_env = np.clip(action_env, -1, 1)
        # #将速度映射到[0,0.6],角速度映射到[-0.9，0.9]
        # action_env[0] = (action_env[0] + 1.0) * 0.3
        # action_env[1] = 0.9 * (action_env[1] + 1.0) - 0.9
        # action_env = [tuple(action_env)]
        # run training update
        if step >= args.init_steps and step % 10 == 0:
            agent.update(replay_buffer, L, step)

        curr_reward = reward
        if step % 500 == 0:
            print("action_env:{}".format(action_env))
        #next_obs, reward, done, _ = env.step(action)
        temp_state = env.step(action_env)

        for i in range(cfg['agent_num_per_env']):
            next_obs[i] = temp_state[0][i]
            next_vector_state[i] = temp_state[1][i]
            reward[i] = temp_state[2][i]
            done[i] = bool(temp_state[3][i])
            is_arrive[i] = bool(temp_state[4]['arrive'][i])

            episode_reward[i] += reward[i]

            replay_buffer.add(obs[i], vector_state[i], action[i], curr_reward[i], reward[i], next_obs[i], next_vector_state[i], done[i])

            obs[i] = next_obs[i]
            vector_state[i] = next_vector_state[i]
        
        flag = 0
        for i in range(cfg['agent_num_per_env']):
            if done[i] == False:
                flag = 1
                break
        if flag == 0:
            global_done = True
        else:
            global_done = False


if __name__ == '__main__':
    main()
