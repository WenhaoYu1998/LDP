import math
import random
from turtle import position
import yaml
import time
from envs import make_env, read_yaml
from prm_astar import get_global_path
# from astar import astar_search
from Astar import astar_search
import json

import numpy as np

import matplotlib.pyplot as plt


class RandomPolicy4Nav:
    def __init__(self, n, v_range=(-0.6, 0.6), w_range=(-0.9, 0.9)):
        self.n = n
        self.v_range = v_range
        self.w_range = w_range

    def gen_action(self):
        out = []

        for i in range(self.n):
            out.append( (random.uniform(*self.v_range), random.uniform(*self.w_range), 0) )

        return out

def convert_to_tuple(lst):
    return tuple(lst)

def astar_interface(start_pose, target_pose, obs_msg):
    obs_list = []
    for obs in obs_msg:
        position = [obs['position'][0], obs['position'][1]]
        radius = obs['size'][2] if obs['shape'] == 'circle' else math.sqrt(obs['size'][0]**2+obs['size'][2]**2)
        obs_list.append(position + [radius,])
        
    yaw = math.atan2(target_pose[1] - start_pose[1], target_pose[0] - start_pose[0]) % (2*math.pi)
    
    start_pose = start_pose + [yaw,]
    target_pose = target_pose + [yaw,]
    astar_path = get_global_path(start_pose, target_pose, obs_list, cfg['robot']['size'][0][0]**2+cfg['robot']['size'][0][2]**2, 20, 5, (16, 16), 0.2)
    return astar_path

if __name__ == "__main__":
    import sys
    tmp = len(sys.argv)
    if tmp == 2:
        cfg = read_yaml(sys.argv[1])
    else:
        cfg = read_yaml('envs/cfg/circle.yaml')
    print(cfg)
    env = make_env(cfg)
    # env2 = make_env(cfg)
    # time.sleep(1)
    random_policy = RandomPolicy4Nav(env.robot_total)
    # test continuous action
    done = True
    episode = 0
    for i in range(1000):
        if done:
            state = env.reset()
            vector = state[1][0]
            target_pose = tuple(state[2][0])
            start_pose = tuple(state[3][0][:2])
            obs_msg = state[4]
            global_map = state[5].reshape(state[5].shape[1], state[5].shape[2])
            target_pose_visual = state[6][0]
            start_pose_visual = state[7][0]
            robot_msg = state[8]
            print(robot_msg)
            # print("start_pose_visual:{}".format(start_pose_visual))
            # print("target_pose_visual:{}".format(target_pose_visual))
            astar_path = astar_search(global_map, start_pose_visual, target_pose_visual)
            # print(astar_path)
            
            # globaldd = cv.resize(global_map,dsize=(160, 160))
            # from PIL import Image
            # # 使用 PIL 创建一个图像
            # img = Image.fromarray(globaldd)
            # # 将图像保存到文件
            # img.save("output.png")
            # exit()
            
            # astar_path = astar_interface(start_pose, target_pose, obs_msg)
            # path_x, path_y = interpolate_curve(astar_path, 3, 500)
            
        temp_state = env.step([(0.6, 0, [], [], astar_path[0], astar_path[1])])
        vector = temp_state[0][1][0]
        target_pose = tuple(temp_state[0][2][0])
        start_pose = tuple(temp_state[0][3][0][:2])
        # obs_msg = temp_state[0][4]
        target_pose_visual = temp_state[0][6][0]
        start_pose_visual = temp_state[0][7][0]
        # print("step_vec:{}".format(vector))
        # print("step_target_pose:{}".format(target_pose))
        # print("step_start_pose:{}".format(start_pose))
        # print("step_target_pose_visual:{}".format(target_pose_visual))
        # print("step_start_pose_visual:{}".format(start_pose_visual))
        obs_msg = temp_state[0][4]
        robot_msg = temp_state[0][8]
        print("&&&&")
        print(robot_msg)
        
        global_map = temp_state[0][5].reshape(temp_state[0][5].shape[1], temp_state[0][5].shape[2])
        done = temp_state[2][0]
        # if done:
        #     if not temp_state[3]['arrive'][0]:
        #         print("Failure: {}".format(2 * episode + 1))
        #         with open(str(2 * episode + 1) + '_failure_obs_list.json', 'w') as file:
        #             json.dump(obs_msg, file)
        #     episode += 1
        astar_path = astar_search(global_map, start_pose_visual, target_pose_visual)
        # astar_path = astar_interface(start_pose, target_pose, obs_msg)
        # path_x, path_y = interpolate_curve(astar_path, 3, 500)
        # print(path_x)
        # print(path_y)
        # plt.figure(figsize=(16, 16))
        # ax = plt.gca()
        # # 设置图表属性
        # plt.xlim(0, 16)
        # plt.ylim(0, 16)
        # # 设置坐标轴为相等，以确保每个单元格都是正方形, 表格横纵坐标比例不一致会造成可视化的矩形为平行四边形
        # plt.axis('equal')
        # plt.title("Global Path")
        # plt.xlabel("X axis")
        # plt.ylabel("Y axis")
        # plt.grid(True)
        # plt.plot(path_x, path_y, 'r.')
        # plt.plot(start_pose[0], start_pose[1], 'bo')
        # plt.savefig("pose_tt.png")
        # print(astar_path)
