# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from Astar import astar_search
import tf
import copy
from PIL import Image


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, vector_state_shape, action_shape, capacity, batch_size, device, is_framestack=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.is_framestack = is_framestack

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32
        vector_state_dtype = np.float32
        if self.is_framestack:
            self.obses = np.empty((capacity,) + (obs_shape[0] + 3,) + obs_shape[1:], dtype=obs_dtype)
            self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        else:
            self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.vector_states = np.empty((capacity, *vector_state_shape), dtype=vector_state_dtype)
            self.k_vector_states = np.empty((capacity, *vector_state_shape), dtype=vector_state_dtype)
            self.next_vector_states = np.empty((capacity, *vector_state_shape), dtype=vector_state_dtype)

        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, vector_state, action, curr_reward, reward, next_obs, next_vector_state, done):
        if self.is_framestack:
            np.copyto(self.obses[self.idx][:-3], obs)
            np.copyto(self.obses[self.idx][-3:], next_obs[-3:])
        else:
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.next_obses[self.idx], next_obs)
            np.copyto(self.vector_states[self.idx], vector_state)
            np.copyto(self.next_vector_states[self.idx], next_vector_state)

        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        if self.is_framestack:
            obses = torch.as_tensor(self.obses[idxs][:, :-3], device=self.device).float()
            next_obses = torch.as_tensor(
                self.obses[idxs][:, 3:], device=self.device
            ).float()
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(
                self.next_obses[idxs], device=self.device
            ).float()
            vector_states = torch.as_tensor(self.vector_states[idxs], device=self.device).float()
            next_vector_states = torch.as_tensor(self.next_vector_states[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, vector_states, actions, rewards, next_obses, next_vector_states, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device), torch.as_tensor(self.k_vector_states[idxs], device=self.device)
        return obses, vector_states, actions, curr_rewards, rewards, next_obses, next_vector_states, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.vector_states[self.last_save:self.idx],
            self.next_vector_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.vector_states[start:end] = payload[2]
            if not self.is_framestack:
                self.next_obses[start:end] = payload[1]
                self.next_vector_states[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.curr_rewards[start:end] = payload[6]
            self.not_dones[start:end] = payload[7]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, cfg, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self.cfg = cfg
        self._frames_obs = []
        self._frames_vector = []
        self._frames_vector_global = []
        self.episode_goal_yaw = 0
        for _ in range(cfg['agent_num_per_env']):
            self._frames_obs.append(deque([], maxlen=k))
            self._frames_vector.append(deque([], maxlen=k))
            self._frames_vector_global.append(deque([], maxlen=k))

    def _create_tf(self, x, y, z):
        q = tf.transformations.quaternion_from_euler(0, 0, z)
        tf_convert = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix((x, y, 0)), 
                                                       tf.transformations.quaternion_matrix(q))
        return tf_convert
    
    def _inverse_tf(self, ogrin_tf):
        return tf.transformations.inverse_matrix(ogrin_tf)
        
    def _global_path_convert(self, global_x, global_y, start_pose_x, start_pose_y, start_pose_yaw):
        global_world_x = global_x / 1066.0 * 16
        global_world_y = global_y / 1066.0 * 16
        tf_base_world = self._create_tf(start_pose_x, start_pose_y, start_pose_yaw)
        tf_target_world = self._create_tf(global_world_x, global_world_y, self.episode_goal_yaw)
        tf_world_target = self._inverse_tf(tf_target_world)
        tf_target_base = self._inverse_tf(np.dot(tf_world_target, tf_base_world)) 
        path_x = tf_target_base[0, 3]
        path_y = tf_target_base[1, 3]
        return path_x, path_y
    
    def reset(self):
        observation = self.env.reset()
        for i in range(self.cfg['agent_num_per_env']):
            vector_state = np.array(observation[1][i]).reshape(1, self.cfg['state_batch'] * self.cfg['state_dim'])
            vector_state_global = copy.deepcopy(vector_state)
            obs = np.array(observation[0][i])
            target_pose = observation[2]
            start_pose = observation[3]
            self.episode_goal_yaw = start_pose[i][2]
            global_map = observation[4].reshape(1, observation[4][i].shape[0], observation[4][i].shape[1])
            resize_global_map = copy.deepcopy(global_map)
            resize_global_map = np.squeeze(resize_global_map).astype(np.uint8)
            global_map_image = Image.fromarray(resize_global_map).convert('L')
            global_map_image = global_map_image.resize((160,160))
            global_map_image = np.array(global_map_image)
            target_pose_visual = observation[5]
            start_pose_visual = observation[6]
            obs_list = observation[7]
            astar_path = astar_search(global_map[i], start_pose_visual[i], target_pose_visual[i])
            if astar_path != [[0],[0]]:
                local_goal_x, local_goal_y = astar_path[0][20], astar_path[1][20]
                if vector_state[i][0] ** 2 + vector_state[i][1] ** 2 > 16:
                    local_goal_world_x = local_goal_x / 1066.0 * 16
                    local_goal_world_y = local_goal_y / 1066.0 * 16
                    tf_base_world = self._create_tf(start_pose[i][0], start_pose[i][1], start_pose[i][2])
                    tf_target_world = self._create_tf(local_goal_world_x, local_goal_world_y, self.episode_goal_yaw)
                    tf_world_target = self._inverse_tf(tf_target_world)
                    tf_target_base = self._inverse_tf(np.dot(tf_world_target, tf_base_world)) 
                    vector_state_global[i][0] = tf_target_base[0, 3]
                    vector_state_global[i][1] = tf_target_base[1, 3]
            for _ in range(self._k):
                self._frames_obs[i].append(obs)
                self._frames_vector[i].append(vector_state)
                self._frames_vector_global[i].append(vector_state_global)
            vector_path_x, vector_path_y = [], []
            if astar_path != [[0], [0]]:
                for p in range(len(astar_path[0])):
                    x, y = self._global_path_convert(astar_path[0][p], astar_path[1][p], start_pose[i][0], start_pose[i][1], start_pose[i][2])
                    vector_path_x.append(x)
                    vector_path_y.append(y)
                vector_path = [vector_path_x, vector_path_y]
            else:
                vector_path = astar_path
            
        return self._get_obs(), self._get_vector(), target_pose, start_pose, global_map, target_pose_visual, start_pose_visual, obs_list, astar_path, self._get_vector_global(), vector_path, global_map_image

    def step(self, action):
        reward, done = [], []
        temp_state = self.env.step(action)
        
        for i in range(self.cfg['agent_num_per_env']):
            obs = np.array(temp_state[0][0][i])
            vector_state = np.array(temp_state[0][1][i]).reshape(1, self.cfg['state_batch'] * self.cfg['state_dim'])
            vector_state_global = copy.deepcopy(vector_state)
            target_pose = temp_state[0][2]
            start_pose = temp_state[0][3]
            global_map = temp_state[0][4].reshape(1, temp_state[0][4][i].shape[0], temp_state[0][4][i].shape[1])
            resize_global_map = copy.deepcopy(global_map)
            resize_global_map = np.squeeze(resize_global_map).astype(np.uint8)
            global_map_image = Image.fromarray(resize_global_map).convert('L')
            global_map_image = global_map_image.resize((160,160))
            global_map_image = np.array(global_map_image)
            target_pose_visual = temp_state[0][5]
            start_pose_visual = temp_state[0][6]
            obs_list = temp_state[0][7]
            reward.append(temp_state[1][i])
            done.append(bool(temp_state[2][i]))
            astar_path = astar_search(global_map[i], start_pose_visual[i], target_pose_visual[i])
            if astar_path != [[0],[0]]:
                local_goal_x, local_goal_y = astar_path[0][20], astar_path[1][20]
                if vector_state[i][0] ** 2 + vector_state[i][1] ** 2 > 16:
                    local_goal_world_x = local_goal_x / 1066.0 * 16
                    local_goal_world_y = local_goal_y / 1066.0 * 16
                    tf_base_world = self._create_tf(start_pose[i][0], start_pose[i][1], start_pose[i][2])
                    tf_target_world = self._create_tf(local_goal_world_x, local_goal_world_y, self.episode_goal_yaw)
                    tf_world_target = self._inverse_tf(tf_target_world)
                    tf_target_base = self._inverse_tf(np.dot(tf_world_target, tf_base_world)) 
                    vector_state_global[i][0] = tf_target_base[0, 3]
                    vector_state_global[i][1] = tf_target_base[1, 3]
            
            self._frames_obs[i].append(obs)
            self._frames_vector[i].append(vector_state)
            self._frames_vector_global[i].append(vector_state_global)
            vector_path_x, vector_path_y = [], []
            if astar_path != [[0], [0]]:
                for p in range(len(astar_path[0])):
                    x, y = self._global_path_convert(astar_path[0][p], astar_path[1][p], start_pose[i][0], start_pose[i][1], start_pose[i][2])
                    vector_path_x.append(x)
                    vector_path_y.append(y)
                vector_path = [vector_path_x, vector_path_y]
            else:
                vector_path = astar_path
            
        info = temp_state[3]
        
        return self._get_obs(), self._get_vector(), reward, done, target_pose, start_pose, global_map, target_pose_visual, start_pose_visual, info, obs_list, astar_path, self._get_vector_global(), vector_path, global_map_image

    def _get_obs(self):
        temp_list_obs = []
        for i in range(self.cfg['agent_num_per_env']):
            assert len(self._frames_obs[i]) == self._k
            temp_list_obs.append(np.concatenate(list(self._frames_obs[i]), axis=0))
        return temp_list_obs
    
    def _get_vector(self):
        temp_list_vector = []
        for i in range(self.cfg['agent_num_per_env']):
            assert len(self._frames_vector[i]) == self._k
            temp_list_vector.append(np.concatenate(list(self._frames_vector[i]), axis=0))
        return temp_list_vector
    
    def _get_vector_global(self):
        temp_list_vector = []
        for i in range(self.cfg['agent_num_per_env']):
            assert len(self._frames_vector_global[i]) == self._k
            temp_list_vector.append(np.concatenate(list(self._frames_vector_global[i]), axis=0))
        return temp_list_vector
