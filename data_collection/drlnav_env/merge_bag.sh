#!/bin/bash
# 临时主题列表文件
TOPICS_FILE="topics.txt"
# 输出 bag 文件
OUTPUT_BAG="merged_static24.bag"
# 包含 bag 文件的目录
BAG_DIRECTORY="src/drl_nav/img_env/cfg_bag/failure_bag"
# 生成所有主题的列表
rosbag info ${BAG_DIRECTORY}/*.bag | grep 'topics:' -A 1000 | grep -oE '/[^ ]+' | sort | uniq > $TOPICS_FILE
# 记录一个包含所有主题的新 bag 文件
rosbag record -O ${OUTPUT_BAG} -a __name:=bag_record &
# 稍等片刻以开始录制
sleep 2
# 播放每个 bag 文件
for BAG_FILE in ${BAG_DIRECTORY}/*.bag
do
    rosbag play $BAG_FILE
done
# 停止录制
rosnode kill /bag_record
# 清理
rm $TOPICS_FILE
