from scipy.spatial.transform import Rotation as R
import numpy as np
import json
import os

def resize_yaw(yaw):
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return [yaw,]

folder_path = './json'

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # 创建字典以按关键词归类
        categories_obs = {
            "obs_position": [],
            "obs_orientation": [],
            "obs_shape": [],
            "obs_size": []
        }

        categories_robot = {
            "robot_init_position": [],
            "robot_orientation": [],
            "robot_goal": []
        }

        # 遍历数据并归类
        for item in data:
            for key in categories_obs.keys():
                if key in item:
                    categories_obs[key].append(item[key])
            for key in categories_robot.keys():
                if key in item:
                    categories_robot[key].append(item[key])

        for i in range(len(categories_obs['obs_position'])):
            # 将四元数转换为欧拉角
            obs_orientation_quaternion = categories_obs['obs_orientation'][i]
            obs_orientation_euler = R.from_quat(obs_orientation_quaternion).as_euler('xyz', degrees=False)
            # 替换原字典中的四元数为欧拉角
            categories_obs['obs_orientation'][i] = [obs_orientation_euler.tolist()]

        for i in range(len(categories_robot['robot_init_position'])):
            # 将四元数转换为欧拉角
            robot_orientation_quaternion = categories_robot['robot_orientation'][i]
            robot_orientation_euler = R.from_quat(robot_orientation_quaternion).as_euler('xyz', degrees=False)
            # 替换原字典中的四元数为欧拉角
            categories_robot['robot_orientation'] = [obs_orientation_euler.tolist()]

        yaml_dict = {
            'robot': [],
            'object': []
        }

        # Robot Yaml
        yaml_dict['robot'].append({'total': len(categories_robot['robot_init_position'])})
        yaml_dict['robot'].append({'shape': ['rectangle'] * len(categories_robot['robot_init_position'])})
        yaml_dict['robot'].append({'size': [[-0.5,0.5,-0.35,0.35]] * len(categories_robot['robot_init_position'])})
        yaml_dict['robot'].append({'begin_poses_type': ['fix'] * len(categories_robot['robot_init_position'])})
        robot_begin_poses = {'begin_poses': []}
        for i in range(len(categories_robot['robot_init_position'])):
            robot_begin_poses['begin_poses'].append(categories_robot['robot_init_position'][i][:2] 
                                                    + resize_yaw(categories_robot['robot_orientation'][i][2]))
        yaml_dict['robot'].append(robot_begin_poses)
        yaml_dict['robot'].append({'target_poses_type': ['fix'] * len(categories_robot['robot_init_position'])})
        robot_target_poses = {'target_poses': []}
        for i in range(len(categories_robot['robot_goal'])):
            robot_target_poses['target_poses'].append(categories_robot['robot_goal'][i])
        yaml_dict['robot'].append(robot_target_poses)

        #Obs Yaml
        yaml_dict['object'].append({'total': len(categories_obs['obs_position'])})
        yaml_dict['object'].append({'shape': categories_obs['obs_shape']})
        obs_size_range = {'size_range': []}
        for i in range(len(categories_obs['obs_position'])):
            if categories_obs['obs_shape'][i] == 'circle':
                obs_size_range['size_range'].append([categories_obs['obs_size'][i][2]] * 2)
            elif categories_obs['obs_shape'][i] == 'rectangle':
                obs_size_range['size_range'].append(categories_obs['obs_size'][i])
        yaml_dict['object'].append(obs_size_range)
        yaml_dict['object'].append({'poses_type': ['fix'] * len(categories_obs['obs_position'])})
        obs_poses = {'poses': []}
        for i in range(len(categories_obs['obs_position'])):
            obs_poses['poses'].append(categories_obs['obs_position'][i][:2] 
                                                    + resize_yaw(categories_obs['obs_orientation'][i][0][2]))
        yaml_dict['object'].append(obs_poses)
        # print(yaml_dict)

        # 手动构建 YAML 格式的字符串
        yaml_str = "robot:\n"
        for item in yaml_dict['robot']:
            for key, value in item.items():
                if isinstance(value, list) and all(isinstance(elem, list) for elem in value):
                    # 处理列表中包含列表的情况
                    yaml_str += f"  {key}: {str(value).replace(' ', '')}\n"
                else:
                    # 处理其他情况
                    yaml_str += f"  {key}: {value}\n"

        yaml_str += "object:\n"
        for item in yaml_dict['object']:
            for key, value in item.items():
                if isinstance(value, list) and all(isinstance(elem, list) for elem in value):
                    # 处理列表中包含列表的情况
                    yaml_str += f"  {key}: {str(value).replace(' ', '')}\n"
                else:
                    # 处理其他情况
                    yaml_str += f"  {key}: {value}\n"

        # 将字符串写入文件
        with open('config_template.yaml', 'r', encoding='utf-8') as read_file:
            with open("./envs/cfg/failure_yaml/" + filename.split('.')[0] + "_config.yaml", "w", encoding="utf-8") as write_file:
                for line in read_file:
                    write_file.write(line)
                write_file.write("init_pose_bag_name: " + filename.split('.')[0] + "_static24.bag\n")
                write_file.write(yaml_str)