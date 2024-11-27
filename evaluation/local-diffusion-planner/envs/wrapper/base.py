from cmath import sqrt
import logging
from envs.env.yaml_env import distance
import gym
import numpy as np
import math
import yaml
import time

from typing import *
from collections import deque
from copy import deepcopy


from envs.state import ImageState
from envs.action import *
from envs.utils import BagRecorder


class StatePedVectorWrapper(gym.ObservationWrapper):
    avg = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0])
    std = np.array([6.0, 6.0, 0.6, 0.9, 0.50, 0.5, 6.0])

    def __init__(self, env, cfg=None):
        super(StatePedVectorWrapper, self).__init__(env)

    def observation(self, state: ImageState):
        self._normalize_ped_state(state.ped_vector_states)
        return state

    def _normalize_ped_state(self, peds):

        for robot_i_peds in peds:
            for j in range(int(robot_i_peds[0])): # j: ped index
                robot_i_peds[1 + j * 7:1 + (j + 1) * 7] = (robot_i_peds[1 + j * 7:1 + (j + 1) * 7] - self.avg) / self.std


class VelActionWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(VelActionWrapper, self).__init__(env)
        if cfg['discrete_action']:
            self.actions: DiscreteActions = DiscreteActions(cfg['discrete_actions'])

            self.f = lambda x: self.actions[int(x)] if np.isscalar(x) else ContinuousAction(*x)
        else:
            self.f = lambda x: ContinuousAction(*x)

    def step(self, action):
        action = self.action(action)
        state, reward, done, info = self.env.step(action)
        info['speeds'] = np.array([a.reverse()[:2] for a in action])
        return state, reward, done, info

    def action(self, actions) -> Iterator[ContinuousAction]:
        return list(map(self.f, actions))

    def reverse_action(self, actions):

        return actions

class AddNoiseWrapper(gym.Wrapper):
    def __init__(self, env, cfg) -> None:
        super(AddNoiseWrapper, self).__init__(env)
        self.mean = cfg['mean']
        self.std = cfg['std']
        self.noise_type = cfg['noise_type']
        self.SNR = cfg['SNR']
    
    def _add_noise(self, obs):
        if self.noise_type == 'salt_pepper':
            SNR = self.SNR
            mask = np.random.choice((0, 1, 2), size=(obs.shape[0], obs.shape[1], obs.shape[2]), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
            mask = np.repeat(mask, obs.shape[0], axis=0)
            obs[mask == 1] = 255
            obs[mask == 2] = 0

        if self.noise_type == 'gaussian':
            obs = obs / 255
            noise = np.random.normal(self.mean, self.std, obs.shape)
            gaussian_out = obs + noise
            gaussian_out = np.clip(gaussian_out, 0, 1)
            gaussian_out = np.float32(gaussian_out * 255)
            obs = gaussian_out
        
        return obs
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        sensor_state = obs.get_sensor_maps()
        sensor_state = self._add_noise(sensor_state)
        obs.change_sensor_maps(sensor_state)
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        sensor_state = state.get_sensor_maps()
        sensor_state = self._add_noise(sensor_state)
        state.change_sensor_maps(sensor_state)
        return state, reward, done, info

class MultiRobotCleanWrapper(gym.Wrapper):
    is_clean : list
    def __init__(self, env, cfg):
        super(MultiRobotCleanWrapper, self).__init__(env)
        self.is_clean = np.array([True] * cfg['agent_num_per_env'])

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        info['is_clean'] = deepcopy(self.is_clean)
        reward[~info['is_clean']] = 0
        info['speeds'][~info['is_clean']] = np.zeros(2)
        # for i in range(len(done)):
        #     if done[i]:
        #         self.is_clean[i]=False
        self.is_clean = np.where(done>0, False, self.is_clean)
        return state, reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.is_clean = np.array([True] * len(self.is_clean))
        return state



class StateBatchWrapper(gym.Wrapper):
    batch_dict: np.ndarray

    def __init__(self, env, cfg):
        print(cfg,flush=True)
        super(StateBatchWrapper, self).__init__(env)
        self.q_sensor_maps = deque([], maxlen=cfg['image_batch']) if cfg['image_batch']>0 else None
        self.q_vector_states = deque([], maxlen=cfg['state_batch']) if cfg['state_batch']>0 else None
        self.q_lasers = deque([], maxlen=cfg['laser_batch']) if cfg['laser_batch']>0 else None
        self.batch_dict = {
            "sensor_maps": self.q_sensor_maps,
            "vector_states": self.q_vector_states,
            "lasers": self.q_lasers,
        }

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.batch_state(state), reward, done, info

    def _concate(self, b: str, t: np.ndarray):
        q = self.batch_dict[b]
        if q is None:
            return t
        else:
            t = np.expand_dims(t, axis=1)
        # start situation
        while len(q) < q.maxlen:
            q.append(np.zeros_like(t))
        q.append(t)
        #  [n(Robot), k(batch), 84, 84]
        return np.concatenate(list(q), axis=1)

    def batch_state(self, state):
        # TODO transpose. print
        state.sensor_maps = self._concate("sensor_maps", state.sensor_maps)
        # print('sensor_maps shape; ', state.sensor_maps.shape)

        # [n(robot), k(batch), state_dim] -> [n(robot), k(batch) * state_dim]
        tmp_ = self._concate("vector_states", state.vector_states)
        state.vector_states = tmp_.reshape(tmp_.shape[0], tmp_.shape[1] * tmp_.shape[2])
        # print("vector_states shape", state.vector_states.shape)


        state.lasers = self._concate("lasers", state.lasers)
        # print("lasers shape:", state.lasers.shape)
        return state

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.batch_state(state)


class SensorsPaperRewardWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(SensorsPaperRewardWrapper, self).__init__(env)

        self.ped_safety_space = cfg['ped_safety_space']
        self.cfg = cfg

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        return states, self.reward(reward, states, info['velocity_a']), done, info

    def _each_r(self, states: ImageState, index: int, velocity):
        distance_reward_factor = 200
        collision_reward = reach_reward = step_reward = distance_reward = rotation_reward = beep_reward = revese_reward =  0

        min_dist = states.ped_min_dists[index]
        vector_state = states.vector_states[index]
        is_collision = states.is_collisions[index]
        is_arrive = states.is_arrives[index]
        step_d = states.step_ds[index]

        if min_dist <= self.ped_safety_space:
            collision_reward = -50 * (self.ped_safety_space - min_dist)
        if is_collision > 0:
            collision_reward = -500.0
        else:
            d = math.sqrt(vector_state[0] ** 2 + vector_state[1] ** 2)
            # print 'robot ',i," dist to goal: ", d
            if d < 0.3 or is_arrive:
                reach_reward = 500.0
            else:
                distance_reward = step_d * distance_reward_factor
                step_reward = -5
        if velocity < 0:
            revese_reward = -10

        reward = collision_reward + reach_reward + step_reward + distance_reward + beep_reward + revese_reward
        return reward

    def reward(self, reward, states, velocity):
        rewards = np.zeros(len(states))
        for i in range(len(states)):
            rewards[i] = self._each_r(states, i, velocity)

        return rewards


class NeverStopWrapper(gym.Wrapper):
    """
        NOTE !!!!!!!!!!!
        put this in last wrapper.
    """
    def __init__(self, env, cfg):
        super(NeverStopWrapper, self).__init__(env)

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        # if info['all_down'][0]:
        #     states = self.env.reset(**info)

        return states, reward, done, info


# time limit
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeLimitWrapper, self).__init__(env)
        self._max_episode_steps = cfg['time_max']
        robot_total = cfg['robot']['total']
        self._elapsed_steps = np.zeros(robot_total, dtype=np.uint8)

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        done = np.where(self._elapsed_steps > self._max_episode_steps, 1, done)
        info['dones_info'] = np.where(self._elapsed_steps > self._max_episode_steps, 10, info['dones_info'])
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class InfoLogWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(InfoLogWrapper, self).__init__(env)
        robot_total = cfg['robot']['total']
        self.tmp = np.zeros(robot_total, dtype=np.uint8)


    def step(self, action):
        states, reward, done, info = self.env.step(action)
        info['arrive'] = states.is_arrives
        info['collision'] = states.is_collisions
        info['refresh_num_episode'] = states.refresh_num_episode
        info['run_dis_episode'] = states.run_dis_episode
        info['run_trajectory_points_episode_x'] = states.run_trajectory_points_episode_x
        info['run_trajectory_points_episode_y'] = states.run_trajectory_points_episode_y

        info['pose'] = states.pose
        info['velocity_a'] = states.velocity_a
        info['target_pose'] = states.target_pose

        info['dones_info'] = np.where(states.is_collisions > 0, states.is_collisions, info['dones_info'])
        info['dones_info'] = np.where(states.is_arrives == 1, 5, info['dones_info'])
        info['all_down'] = self.tmp + sum(np.where(done>0, 1, 0)) == len(done)
        return states, reward, done, info


class BagRecordWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(BagRecordWrapper, self).__init__(env)
        self.bag_recorder = BagRecorder(cfg["bag_record_output_name"])
        self.record_epochs = int(cfg['bag_record_epochs'])
        self.episode_res_topic = "/" + cfg['env_name'] + str(cfg['node_id']) + "/episode_res"
        print("epi_res_topic", self.episode_res_topic, flush=True)
        self.cur_record_epoch = 0

        self.bag_recorder.record(self.episode_res_topic)

    def _trans2string(self, dones_info):
        o: List[str] = []
        for int_done in dones_info:
            if int_done == 10:
                o.append("stuck")
            elif int_done == 5:
                o.append("arrive")
            elif 0 < int_done < 4:
                o.append("collision")
            else:
                raise ValueError
        print(o, flush=True)
        return o

    def reset(self, **kwargs):
        if self.cur_record_epoch == self.record_epochs:
            time.sleep(10)
            self.bag_recorder.stop()
        if kwargs.get('dones_info') is not None: # first reset not need
            self.env.end_ep(self._trans2string(kwargs['dones_info']))
            self.cur_record_epoch += 1
            """
                done info:
                10: timeout
                5:arrive
                1: get collision with static obstacle
                2: get collision with ped
                3: get collision with other robot
            """
        print(self.cur_record_epoch, flush=True)
        return self.env.reset(**kwargs)


class TestEpisodeWrapper(gym.Wrapper):
    """
    for one robot
    calc trajectory length
    """
    def __init__(self, env, cfg):
        super(TestEpisodeWrapper, self).__init__(env)
        self.episode_dis = np.zeros([cfg['agent_num_per_env'], 1])
        self.step_hz_ = cfg['control_hz']
        self.control_hz_ = 0.05
        self.all_traj_len = 0
        self.num = 0
        self.exe_time = np.zeros([cfg['agent_num_per_env'], 1])
        self.run_dis = np.zeros([cfg['agent_num_per_env'], 1])
        self.success = np.zeros([cfg['agent_num_per_env'], 1])
        self.success_num = np.zeros([cfg['agent_num_per_env'], 1])
        self.collision = np.zeros([cfg['agent_num_per_env'], 1])
        self.collision_num = np.zeros([cfg['agent_num_per_env'], 1])
        self.stuck = np.zeros([cfg['agent_num_per_env'], 1])
        self.stuck_num = np.zeros([cfg['agent_num_per_env'], 1])
        self.exe_time_arr = []
        self.dis_len_arr = []
        self.start_time = 0
        self.episode = 0
        self.cfg = cfg
        self.SPL_arr = []
        # self.robot_trajectory_curvature_arr = []
        self.run_trajectory_points_x_arr = []
        self.run_trajectory_points_y_arr = []
        for i in range(cfg['agent_num_per_env']):
            self.exe_time_arr.append(np.zeros([1000, 1]))
            self.dis_len_arr.append(np.zeros([1000, 1]))
            self.SPL_arr.append(np.zeros([1000, 1]))
            # self.robot_trajectory_curvature_arr.append(np.zeros([1000, 1]))

    def _calc_curvature(self, index):
        curvature = 0
        points_num = len(self.run_trajectory_points_x_arr[index])
        for i in range(1, points_num - 1):
            P1_x = self.run_trajectory_points_x_arr[index][i-1]
            P1_y = self.run_trajectory_points_y_arr[index][i-1]
            P2_x = self.run_trajectory_points_x_arr[index][i]
            P2_y = self.run_trajectory_points_y_arr[index][i]
            P3_x = self.run_trajectory_points_x_arr[index][i+1]
            P3_y = self.run_trajectory_points_y_arr[index][i+1]

            P2xP3y = P2_x * P3_y
            P3xP2y = P3_x * P2_y

            P2x_P3x = P2_x - P3_x
            P2y_P3y = P2_y - P3_y

            P1xxpP1yy = P1_x * P1_x + P1_y * P1_y
            P2xxpP2yy = P2_x * P2_x + P2_y * P2_y
            P3xxpP3yy = P3_x * P3_x + P3_y * P3_y

            A = P1_x * P2y_P3y - P1_y * P2x_P3x + P2xP3y - P3xP2y
            B = P1xxpP1yy * (-P2y_P3y) + P2xxpP2yy * (P1_y - P3_y) + P3xxpP3yy * (P2_y - P1_y)
            C = P1xxpP1yy * P2x_P3x + P2xxpP2yy * (P3_x - P1_x) + P3xxpP3yy * (P1_x - P2_x)
            D = P1xxpP1yy * (P3xP2y - P2xP3y) + P2xxpP2yy * (P1_x * P3_y - P3_x * P1_y) + P3xxpP3yy * (P2_x * P1_y - P1_x * P2_y)

            if A == 0:
                points_num -= 1
                continue

            R = math.sqrt((B * B + C * C - 4 * A * D) / (4 * A * A))

            curvature += 1 / R

        if curvature == 0:
            return curvature
        else:
            return curvature / (points_num - 2) # 一个episode所有点的平均曲率

    def calc_dis(self, v, w):
        cur_control = 0.05
        step_dis = 0
        while(cur_control <= self.step_hz_):  
            theta = w * cur_control
            dis_x = v * self.control_hz_ * math.cos(theta)
            dis_y = v * self.control_hz_ * math.sin(theta)
            step_dis += math.sqrt(dis_x * dis_x + dis_y * dis_y)
            cur_control += self.control_hz_
        return step_dis

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        for i in range(self.cfg['agent_num_per_env']):
            if self.exe_time[i] == 0:
                if done[i]:
                    self.success[i] = info['arrive'][i]
                    self.collision[i] = info['collision'][i]
                    if info['arrive'][i] == 0 and info['collision'][i] == 0:
                        self.stuck[i] = 1
                    else:
                        self.stuck[i] = 0
                    self.exe_time[i] = info['refresh_num_episode'][i] * 0.05
                    self.run_dis[i] = info['run_dis_episode'][i]
                    self.run_trajectory_points_x_arr.append(info['run_trajectory_points_episode_x'][i])
                    self.run_trajectory_points_y_arr.append(info['run_trajectory_points_episode_y'][i])
                    self.pose = info['pose']
                    self.velocity_a = info['velocity_a']
                    self.target_pose = info['target_pose']

        return states, reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.start_time = time.time()
        if not np.all(self.exe_time == 0):
            SUM_exe_time = 0
            SUM_run_dis = 0
            SUM_spl = 0
            # SUM_trajectory_curvature = 0
            for i in range(self.cfg['agent_num_per_env']):
                if self.success[i]:
                    self.success_num[i] += 1
                    self.exe_time_arr[i][self.episode % 1000] = self.exe_time[i]
                    self.dis_len_arr[i][self.episode % 1000] = self.run_dis[i]
                    direct_dis = math.sqrt(state.vector_states[0][0]**2 + state.vector_states[0][1]**2)
                    self.SPL_arr[i][self.episode % 1000] = direct_dis / max(self.run_dis[i], direct_dis)
                else:
                    if self.collision[i]:
                        self.collision_num[i] += 1
                    if self.stuck[i]:
                        self.stuck_num[i] += 1
                    self.exe_time_arr[i][self.episode % 1000] = 0
                    self.dis_len_arr[i][self.episode % 1000] = 0
                    self.SPL_arr[i][self.episode % 1000] = 0
                
                print("robot_" + str(i) + "_succ_rate: {}".format(self.success_num[i] / (self.episode + 1)))
                print("robot_" + str(i) + "_coll_rate: {}".format(self.collision_num[i] / (self.episode + 1)))
                print("robot_" + str(i) + "_stuck_rate: {}".format(self.stuck_num[i] / (self.episode + 1)))
                robot_exe_time = np.sum(self.exe_time_arr[i]) / self.success_num[i]
                print("robot_" + str(i) + "_exe_time: {}".format(robot_exe_time))
                dis_len = np.sum(self.dis_len_arr[i]) / self.success_num[i]
                print("robot_" + str(i) + "_dis_len: {}".format(dis_len))
                spl = np.sum(self.SPL_arr[i]) / 1000
                print("robot_" + str(i) + "_spl: {}".format(spl))
                # self.robot_trajectory_curvature_arr[i][self.episode % 1000] = self._calc_curvature(i)
                # robot_trajectory_curvature = np.sum(self.robot_trajectory_curvature_arr[i]) / 1000
                # print("robot_" + str(i) + "_trajectory_curvature: {}".format(robot_trajectory_curvature))
            self.episode += 1
            if self.episode > 1000:
                for i in range(self.cfg['agent_num_per_env']):
                    SUM_exe_time += robot_exe_time
                    SUM_run_dis += dis_len
                    SUM_spl += spl
                    # SUM_trajectory_curvature += robot_trajectory_curvature
                print("AVE_exe_time: {}".format(SUM_exe_time / self.cfg['agent_num_per_env']))
                print("AVE_run_dis: {}".format(SUM_run_dis / self.cfg['agent_num_per_env']))
                print("AVE_spl: {}".format(SUM_spl / self.cfg['agent_num_per_env']))
                # print("AVE_trajectory_curvature: {}".format(SUM_trajectory_curvature / self.cfg['agent_num_per_env']))

        self.run_dis[:] = 0
        self.exe_time[:] = 0

        return state