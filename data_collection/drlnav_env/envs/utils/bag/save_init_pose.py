import yaml
import sys
import os
sys.path.append(sys.path[0]+"/../../../")

from envs.env import ImageEnv


def read_yaml(file: str) -> dict:

    file = open(file, 'r', encoding="utf-8")
    # 读取文件中的所有数据
    file_data = file.read()
    file.close()

    # 指定Loader
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data


if __name__ == "__main__":
    folder_path = 'envs/cfg/failure_yaml'

    for filename in os.listdir(folder_path):
        if filename.endswith('.yaml'):
            f = read_yaml(os.path.join(folder_path, filename))
            env = ImageEnv(f)
            env.seed(2)
            env.save_envs_bag(f['init_pose_bag_episodes'], f['init_pose_bag_name'])
