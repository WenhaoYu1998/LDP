#! /usr/bin/env python
from comn_pkg.srv import VFHPlan
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import rospy

path_follower_cmd_pub = rospy.Publisher('/path_follower_cmd', String, None, True, queue_size=100)
def path_follower_cmd(cmd):
    path_follower_cmd_pub.publish(String(cmd))

def forward_move(topic):
    path_follower_cmd("forward " + topic)

def backward_move(topic):
    path_follower_cmd("backward " + topic)

def wait_move(is_open):
    if is_open:
        path_follower_cmd("wait 1")
    else:
        path_follower_cmd("wait 0")

def wait_horizon(x, y):
    path_follower_cmd("wait_horizon {0} {1}".format(x, y))

def blind_move(is_open):
    if is_open:
        path_follower_cmd("blind 1")
    else:
        path_follower_cmd("blind 0")

def reset_path_follower():
    path_follower_cmd("reset")

def vfh_plan(target, scan, robot_namespace=None):
    if robot_namespace != None:
        srv_name = '/' + robot_namespace + '/' + 'vfh_plan'
        laser_link = robot_namespace + '/' + 'laser_link'
    else:
        srv_name = '/vfh_plan'
        laser_link = 'laser_link'
    rospy.wait_for_service('/vfh_planner/vfh_plan')
    try:
        plan = rospy.ServiceProxy('/vfh_planner/vfh_plan', VFHPlan)
        resp = plan(target, scan)
        return [resp.vel.linear.x, resp.vel.angular.z]
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

class CallTest(object):
    """docstring for CallTest"""
    def __init__(self):
        super(CallTest, self).__init__()
        forward_move("/scan")
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback,queue_size=1)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "odom"
        self.target_pose.pose.position.x = 20
        self.target_pose.pose.orientation.w = 1
    def scan_callback(self, msg):
        self.target_pose.header.stamp = msg.header.stamp
        vw = vfh_plan(self.target_pose, msg)
        vel = Twist()
        vel.linear.x = vw[0]
        vel.angular.z = vw[1]
        self.vel_pub.publish(vel)

if __name__ == '__main__':
    rospy.init_node("test_call", anonymous=True)
    test = CallTest()
    rospy.spin()