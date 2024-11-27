# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import tempdir
from tracemalloc import start
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
import pickle
import copy
import utils
from logger import Logger
from Astar import astar_search

from envs import make_env, read_yaml


def parse_args():
    """Parse command line arguments for training&testing configuration"""
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
    parser.add_argument('--delta_w_arr_len', default=1000, type=int)
    parser.add_argument('--preference', default='local', type=str, choices=['local', 'global'])
    parser.add_argument('--output_path', default=None, type=str)
    parser.add_argument('--env_config', default=None, type=str)
    parser.add_argument('--save_episode', default=1000, type=int)

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

def make_agent(obs_shape, vector_state_shape, action_shape, args, device):
    """
    Create agent based on specified type
    Args:
        obs_shape: Shape of observation space
        vector_state_shape: Shape of state vector
        action_shape: Shape of action space
        args: Training arguments
        device: Computing device (CPU/GPU)
    Returns:
        agent: Instantiated agent object
    """
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

    ###
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

    if args.domain_name == 'drl_env':
        # Configure Environment
        cfg = read_yaml(args.env_config)
        env = make_env(cfg)
        env.seed(args.seed)
        eval_env = env
        
    if args.domain_name == 'drl_env':
    # stack several consecutive frames together
        if args.encoder_type.startswith('pixel'):
            env = utils.FrameStack(env, cfg, k=args.frame_stack)
            eval_env = utils.FrameStack(eval_env, cfg, k=args.frame_stack)

    # Create directories for saving outputs
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    # Save configuration parameters
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup replay buffer
    replay_buffer = utils.ReplayBuffer(
        obs_shape=(args.frame_stack, cfg['image_size'][0], cfg['image_size'][1]),
        vector_state_shape=(args.frame_stack, cfg['state_dim'] * cfg['state_batch']),
        action_shape=(2,),
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
    )

    # Initialize logger
    agent = make_agent(
        obs_shape=(cfg['image_batch'], cfg['image_size'][0], cfg['image_size'][1]),
        vector_state_shape=(cfg['state_dim'] * cfg['state_batch'],),
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
    delta_w_arr, traj_len_arr = [], []
    obs, vector_state, next_obs, next_vector_state= [], [], [], []
    global_map, nex_global_map, target_pose_visual, start_pose_visual = [], [], [], []
    target_pose_robot, start_pose_robot = [], []
    last_w, obs_list = [], []
    astar_path, next_astar_path, vector_path, next_vector_path = [], [], [], []
    global_path_goal, next_global_path_goal= [], []
    global_map_image, next_global_map_image = [], []
    for i in range(cfg['agent_num_per_env']):
        episode_reward.append(0.0)
        done.append(True)
        is_arrive.append(False)
        arrive_arr.append(np.zeros([args.arrive_arr_len, 1]))
        delta_w_arr.append(np.zeros([args.delta_w_arr_len, 1]))
        obs.append(0.0)
        vector_state.append(0.0)
        global_map.append(0.0)
        nex_global_map.append(0.0)
        global_map_image.append(0.0)
        next_global_map_image.append(0.0)
        target_pose_visual.append(0.0)
        start_pose_visual.append(0.0)
        target_pose_robot.append(0.0)
        start_pose_robot.append(0.0)
        obs_list.append(0.0)
        next_obs.append(0.0)
        next_vector_state.append(0.0)
        last_w.append(0.0)
        astar_path.append(0.0)
        next_astar_path.append(0.0)
        vector_path.append(0.0)
        next_vector_path.append(0.0)
        global_path_goal.append(0.0)
        next_global_path_goal.append(0.0)

    start_time = time.time()
    global_done = True
    model_path = os.path.dirname(args.load_encoder)
    model_id = os.path.basename(args.load_encoder).split('_')[1].split('.')[0]
    agent.load(model_path, model_id)

    pre_pose = np.array([])
    target_pose = np.array([])
    
    save_episode_num = 0
    episode_data_list = []

    for step in range(args.num_train_steps):
        if global_done:
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)
            
            for i in range(cfg['agent_num_per_env']):
                if is_arrive[i]:
                    arrive_arr[i][episode % args.arrive_arr_len] = 1
    
                elif time.time() - start_time == 200:
                    arrive_arr[i][episode % args.arrive_arr_len] = 0
                else:
                    arrive_arr[i][episode % args.arrive_arr_len] = 0
                
                L.log('eval/episode_reward_robot_' + str(i), episode_reward[i], step)

            # Reset environment for the next episode
            done, episode_reward, reward = [], [], []
            observation = env.reset()

            pre_pose = observation[3]
            target_pose = observation[2]
            
            for i in range(cfg['agent_num_per_env']):
                obs[i] = np.array(observation[0][i])
                vector_state[i] = np.array(observation[1][i])
                target_pose_robot[i] = observation[2][i]
                start_pose_robot[i] = observation[3][i]
                global_map[i] = observation[4][i]
                target_pose_visual[i] = observation[5][i]
                start_pose_visual[i] = observation[6][i]
                obs_list[i] = observation[7]
                done.append(False)
                episode_reward.append(0.0)
                reward.append(0.0)
                last_w[i] = 0.0
                arrive_rate = np.sum(arrive_arr[i]) / args.arrive_arr_len
                delta_w_rate = np.sum(delta_w_arr[i]) / args.delta_w_arr_len
                astar_path[i] = observation[8]
                global_path_goal[i] = np.array(observation[9][i])
                vector_path[i] = observation[10]
                global_map_image[i] = observation[11]
                
            episode += 1
            L.log('eval/episode', episode, step)
            if save_episode_num >= args.save_episode:
                exit()
       
        action, action_env = [], []
        for i in range(cfg['agent_num_per_env']):
            with utils.eval_mode(agent):
                # Choose action based on preference (local/global)
                if args.preference == 'local':
                    action.append(agent.sample_action(obs[i], vector_state[i]))
                elif args.preference == 'global':
                    action.append(agent.sample_action(obs[i], global_path_goal[i]))

            action_env.append(action[i] * 1.1)
            action_env[i] = np.clip(action_env[i], -1, 1)
            action_env[i][0] = 0.6 * (action_env[i][0] + 1.0) - 0.6
            action_env[i][1] = 0.785 * (action_env[i][1] + 1.0) - 0.785
            action_env[i] = tuple(action_env[i])
            action_env[i] = action_env[i] + ([], [], astar_path[i][0], astar_path[i][1],)
            delta_w_arr[i][step % args.delta_w_arr_len] = abs(action_env[i][1] - last_w[i])
            
        temp_state = env.step(action_env)

        for i in range(cfg['agent_num_per_env']):

            next_obs[i] = temp_state[0][i]
            next_vector_state[i] = temp_state[1][i]
            reward[i] = temp_state[2][i]
            done[i] = bool(temp_state[3][i])
            nex_global_map[i] = temp_state[6][i]
            target_pose_visual[i] = temp_state[7][i]
            start_pose_visual[i] = temp_state[8][i]
            is_arrive[i] = bool(temp_state[9]['arrive'][i])
            obs_list[i] = temp_state[10]
            episode_reward[i] += reward[i]
            next_astar_path[i] = temp_state[11]
            next_global_path_goal[i] = temp_state[12][i]
            next_vector_path[i] = temp_state[13]
            next_global_map_image[i] = temp_state[14]
            
            # Store current round data
            obs_data = copy.deepcopy(obs[i])
            vector_state_data = copy.deepcopy(vector_state[i])
            global_astar_path_data = copy.deepcopy(np.array(astar_path[i]))
            global_vector_path_data = copy.deepcopy(np.array(vector_path[i]))
            global_map_data = copy.deepcopy(np.array(global_map_image[i]))

            obs[i] = next_obs[i]
            vector_state[i] = next_vector_state[i]
            astar_path[i] = next_astar_path[i]
            vector_path[i] = next_vector_path[i]
            global_path_goal[i] = next_global_path_goal[i]
            global_map[i] = nex_global_map[i]
            global_map_image[i] = next_global_map_image[i]
            last_w[i] = action_env[i][1]

            # save data
            action_data = copy.deepcopy(np.array(action_env[i][:2]))
            reward_data = copy.deepcopy(temp_state[2][i])
            pose_data = copy.deepcopy(pre_pose)
            target_pose_data = copy.deepcopy(target_pose)

            velocity_data = temp_state[9]['velocity_a']
            pre_pose = temp_state[9]['pose']

            done_data = copy.deepcopy(temp_state[3][i])

            # Determine end condition
            end_condition = 0
            if done_data and temp_state[9]['arrive'][i]==False and temp_state[9]['collision'][i]==False:
                end_condition = 2
                #end_condition = 'Exhausted'
            elif temp_state[9]['arrive'][i]:
                end_condition = 1
                #end_condition = 'Arrived'
            elif temp_state[9]['collision'][i]:
                end_condition = 3
                #end_condition = 'Collision'
            
            # Data structure to save
            data_dict = {
            'obs_data': obs_data[-1],
            'vector_state': vector_state_data[-1],
            'action_data': action_data,
            'global_astar_path': global_astar_path_data,
            'global_vector_path': global_vector_path_data,
            'global_map': global_map_data,
            'reward_data': reward_data,
            'pose_data': pose_data,
            'done': done_data,
            'end_condition': end_condition,
            'target_pose': target_pose_data,
            'velocity':velocity_data
            }
            
            episode_data_list.append(data_dict)
            file_path = args.output_path

            # Save successfully completed episode data
            if end_condition == 1:
                for data in episode_data_list:
                    if os.path.exists(file_path):
                        with open(file_path, 'ab') as f:
                            pickle.dump(data, f)
                    else:
                        with open(file_path, 'wb') as f:
                            pickle.dump(data, f)
                save_episode_num += 1
                print("num: {}".format(save_episode_num))
                episode_data_list.clear()
            elif end_condition == 2 or end_condition == 3:
                episode_data_list.clear()
                with open('./video/' + str(episode) + '_failure_obs_list.json', 'w') as file:
                    json.dump(obs_list[i], file)
            
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
