import tf
import rospy
import numpy as np
from geometry_msgs.msg import Pose

# 示例用法
base_pose = Pose()
base_pose.position.x = 4.11832587 
base_pose.position.y = 11.72676396
base_pose.orientation.z = 1.18900081 # 假设这是以弧度为单位的旋转

base_q = tf.transformations.quaternion_from_euler(0, 0, base_pose.orientation.z)
tf_base_world = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix((base_pose.position.x, base_pose.position.y, 0)), 
                                                       tf.transformations.quaternion_matrix(base_q)) 

target_pose = Pose()
target_pose.position.x = 12.78448341
target_pose.position.y = 3.35940581
target_pose.orientation.z = 1.18900081

target_q = tf.transformations.quaternion_from_euler(0, 0, target_pose.orientation.z)
tf_target_world = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix((target_pose.position.x, target_pose.position.y, 0)), 
                                                       tf.transformations.quaternion_matrix(target_q)) 

tf_world_target = tf.transformations.inverse_matrix(tf_target_world)

tf_target_base = tf.transformations.inverse_matrix(np.dot(tf_world_target , tf_base_world))

vector_state_x = tf_target_base[0, 3]
vector_state_y = tf_target_base[1, 3]
print(vector_state_x, vector_state_y)

