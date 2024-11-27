"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

from operator import le
from random import seed
from re import X
import sys
import numpy as np
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import time
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

from envs import make_env, read_yaml
from collections import deque
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from Astar import astar_search
import tf
import copy
from PIL import Image

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    # ibc, gmm需要注释掉
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    # env_runner = hydra.utils.instantiate(
    #     cfg.task.env_runner,
    #     output_dir=output_dir)
    # env_runner = NavImageRunner(output_dir=output_dir)
    # runner_log = env_runner.run(policy)
    
    env_cfg = read_yaml('envs/cfg/circle.yaml')
    env = make_env(env_cfg)
    env.seed(2)
    To = 2
    episode = 0
    done = True
    frame_obs = deque([], maxlen=To) # maxlen == To
    frame_vector = deque([], maxlen=To)
    frame_global_path = deque([], maxlen=To)
    # frame_global_map = deque([], maxlen=To)
    arrive_num = 0
    np_obs_dict = {}
    for step in range(1000000):
        if done:
            observation = env.reset()
            policy.reset()
            obs = np.array(observation[0][0]).reshape(1, 1, 84, 84)
            vector_state = np.array(observation[1][0]).reshape(1, 3)
            start_pose = observation[3]
            episode_goal_yaw = start_pose[0][2]
            global_map = observation[4].reshape(observation[4].shape[1], observation[4].shape[2])
            # resize_global_map = copy.deepcopy(global_map)
            # resize_global_map = np.squeeze(resize_global_map).astype(np.uint8)
            # global_map_image = Image.fromarray(resize_global_map).convert('L')
            # global_map_image = global_map_image.resize((160,160))
            # global_map_image = np.array(global_map_image).reshape(1, 1, 160, 160)
            target_pose_visual = observation[5][0]
            start_pose_visual = observation[6][0]
            # astar_path = astar_search(global_map, start_pose_visual, target_pose_visual)
            astar_path = [[0] * 100, [0] * 100]
            vector_path_x, vector_path_y = [], []
            if astar_path != [[0] * 100, [0] * 100]:
                for p in range(len(astar_path[0])):
                    x, y = global_path_convert(astar_path[0][p], astar_path[1][p], start_pose[0][0], start_pose[0][1], start_pose[0][2], episode_goal_yaw)
                    vector_path_x.append(x)
                    vector_path_y.append(y)
                vector_path = [vector_path_x, vector_path_y]
            else:
                vector_path = [[x] * 100 for x in vector_state[0][:2]]
            
            for _ in range(To):
                frame_obs.append(obs)
                frame_vector.append(vector_state)
                frame_global_path.append(np.array(vector_path).T[[20, 40, 60, 80], :].tolist())
                # frame_global_map.append(global_map_image)

            np_obs_dict.clear()
            np_obs_dict["agent_pos"] = np.concatenate(list(frame_vector), axis=0).reshape(1, To, 3) # B, To, ...
            np_obs_dict["image"] = np.concatenate(list(frame_obs), axis=0).reshape(1, To, 1, 84, 84)/255
            np_obs_dict["global_path"] = np.concatenate(list(frame_global_path), axis=0).reshape(1, To, 4, 2)
            # np_obs_dict["global_map"] = np.concatenate(list(frame_global_map), axis=0).reshape(1, To, 1, 160, 160)/255
            print("episode:{}".format(episode))
            if episode > 1001:
                exit()
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))
        # images = obs_dict['image'][0]
        # # 设置绘图
        # fig, axes = plt.subplots(2, 2, figsize=(10, 10)) # 创建一个 4x4 的子图网格
        # axes = axes.ravel()  # 将子图数组转换为一维数组，便于迭代
        # # 遍历并绘制每个图像
        # for i in range(To):
        #     axes[i].imshow(images[i].squeeze().cpu(), cmap='gray')  # 绘制图像，使用灰度色图
        #     axes[i].set_title(f'#{i+1}')
        #     axes[i].axis('off')  # 关闭坐标轴显示
        # plt.savefig("batch.png")  # 显示图像
        # plt.show()
        
        start_time = time.time()
        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)
        end_time = time.time()
        # print("A action step time:{}".format(end_time - start_time))
        
        action = action_dict["action"][0, :, :].tolist()
        # action_pred = action_dict["action_pred"][0, :, :].tolist()
        # print("action:{}".format(action))
        action_v, action_w, action_pred_v, action_pred_w = [], [], [], []
        for a in action:
            action_v.append(a[0])
            action_w.append(a[1])
        # for a in action_pred[(To-1):]:
        #     action_pred_v.append(a[0])
        #     action_pred_w.append(a[1])
        # print("action_v:{}".format(action_v))
        # print("action_w:{}".format(action_w))
        for i in range(len(action_v[:4])):
            # action = [(action_v[i], action_w[i], action_v, action_w)]
            if action_v[i] > 0.6:
                action_v[i] = 0.6
            if action_v[i] < -0.6:
                action_v[i] = -0.6
            if action_w[i] > 0.785:
                action_w[i] = 0.785
            if action_w[i] < -0.785:
                action_w[i] = -0.785
            action = [(action_v[i], action_w[i], action_pred_v, action_pred_w, astar_path[0], astar_path[1])]
            temp_state = env.step(action)
        
            next_obs = np.array(temp_state[0][0][0]).reshape(1, 1, 84, 84)
            next_vector_state = np.array(temp_state[0][1][0]).reshape(1, 3)
            start_pose = temp_state[0][3]
            global_map = temp_state[0][4].reshape(temp_state[0][4].shape[1], temp_state[0][4].shape[2])
            # resize_global_map = copy.deepcopy(global_map)
            # resize_global_map = np.squeeze(resize_global_map).astype(np.uint8)
            # global_map_image = Image.fromarray(resize_global_map).convert('L')
            # global_map_image = global_map_image.resize((160,160))
            # global_map_image = np.array(global_map_image).reshape(1, 1, 160, 160)
            target_pose_visual = temp_state[0][5][0]
            start_pose_visual = temp_state[0][6][0]
            # astar_path = astar_search(global_map, start_pose_visual, target_pose_visual)
            astar_path = [[0] * 100, [0] * 100]
            vector_path_x, vector_path_y = [], []
            if astar_path != [[0] * 100, [0] * 100]:
                for p in range(len(astar_path[0])):
                    x, y = global_path_convert(astar_path[0][p], astar_path[1][p], start_pose[0][0], start_pose[0][1], start_pose[0][2], episode_goal_yaw)
                    vector_path_x.append(x)
                    vector_path_y.append(y)
                vector_path = [vector_path_x, vector_path_y]
            else:
                vector_path = [[x] * 100 for x in next_vector_state[0][:2]]
            
            frame_obs.append(next_obs)
            frame_vector.append(next_vector_state)
            frame_global_path.append(np.array(vector_path).T[[20, 40, 60, 80], :].tolist())
            # frame_global_map.append(global_map_image)
            np_obs_dict["agent_pos"] = np.concatenate(list(frame_vector), axis=0).reshape(1, To, 3) # B, To, ...
            np_obs_dict["image"] = np.concatenate(list(frame_obs), axis=0).reshape(1, To, 1, 84, 84)/255
            np_obs_dict["global_path"] = np.concatenate(list(frame_global_path), axis=0).reshape(1, To, 4, 2)
            # np_obs_dict["global_map"] = np.concatenate(list(frame_global_map), axis=0).reshape(1, To, 1, 160, 160)/255
        
            reward = temp_state[1][0]
            done = bool(temp_state[2][0])
            
            if done:
                episode = episode + 1
            if temp_state[3]['arrive'][0]:
                arrive_num += 1
                print("arrive_rate:{}".format(arrive_num/episode))
            if done:
                break
        
    
    # # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

def global_path_convert(global_x, global_y, start_pose_x, start_pose_y, start_pose_yaw, episode_goal_yaw):
        global_world_x = global_x / 1066.0 * 16
        global_world_y = global_y / 1066.0 * 16
        tf_base_world = create_tf(start_pose_x, start_pose_y, start_pose_yaw)
        tf_target_world = create_tf(global_world_x, global_world_y, episode_goal_yaw)
        tf_world_target = inverse_tf(tf_target_world)
        tf_target_base = inverse_tf(np.dot(tf_world_target, tf_base_world)) 
        path_x = tf_target_base[0, 3]
        path_y = tf_target_base[1, 3]
        return path_x, path_y

def create_tf(x, y, z):
        q = tf.transformations.quaternion_from_euler(0, 0, z)
        tf_convert = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix((x, y, 0)), 
                                                       tf.transformations.quaternion_matrix(q))
        return tf_convert
    
def inverse_tf(ogrin_tf):
    return tf.transformations.inverse_matrix(ogrin_tf)

if __name__ == '__main__':
    main()
