import rosbag
import os

def merge_bags(input_directory, output_bag_file):
    # 创建一个新的 bag 文件，用于存储合并后的数据
    with rosbag.Bag(output_bag_file, 'w') as outbag:
        # 遍历指定目录下的所有 bag 文件
        for bag_file in os.listdir(input_directory):
            if bag_file.endswith('.bag'):
                print(bag_file)
                # 打开当前 bag 文件
                with rosbag.Bag(os.path.join(input_directory, bag_file), 'r') as inbag:
                    # 将消息从当前 bag 文件复制到输出 bag 文件
                    for topic, msg, t in inbag.read_messages():
                        outbag.write(topic, msg, t)

# 使用示例
input_dir = 'src/drl_nav/img_env/cfg_bag/failure_bag'  # 替换为你的 bag 文件所在目录的路径
output_file = 'src/drl_nav/img_env/cfg_bag/merged_static24.bag'  # 替换为你的输出 bag 文件的路径
merge_bags(input_dir, output_file)
