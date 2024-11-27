#include "vfhlite.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <boost/thread.hpp>
#include <comn_pkg/PathFollowAction.h>
#include <comn_pkg/VFHPlan.h>
#include <actionlib/server/simple_action_server.h>
#include <message_filters/subscriber.h>
#include <tf/message_filter.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/String.h>
#include "peek_cmd.h"

class RosLaserData
{
public:
  RosLaserData()
  {
    stamp = ros::Time(0);
    data = NULL;
  }
  ros::Time stamp;
  double* data;
};
class LocalMap
{
public:
    LocalMap(){}
    void init(int size_x, int size_y, float delta)
    {
        delta_ = delta;
        data_ = new int[size_x * size_y];
        size_x_ = size_x;
        size_y_ = size_y;
        tf_map_base_.setOrigin(tf::Vector3(size_x_ * delta_ / 2.0, size_y_ * delta_ / 2.0, 0));
        tf_map_base_.setRotation(tf::Quaternion(0.707388,-0.706825,3.27463e-05,-3.27724e-05));
        tf_base_map_ = tf_map_base_.inverse();
        clear();
    }
    void clear()
    {
        for (int i = 0; i < size_x_ * size_y_; i++)
            data_[i] = 0;
    }
    bool free()
    {
        delete data_;
        data_ = NULL;
    }
    ~LocalMap()
    {
        delete data_;
    }
    int getIndex(float p_x, float p_y)
    {
        tf::Vector3 p_base(p_x, p_y, 0);
        tf::Vector3 p_map = tf_base_map_ * p_base;
        int x = p_map.getX() / delta_;
        int y = p_map.getY() / delta_;
        if (x >= 0 && x < size_y_ && y >= 0 && y < size_x_)
        {
            return (y*size_y_ + x);
        }
        else
            return -1;
    }
    int* data_;
    float delta_;
    int size_x_;
    int size_y_;
    tf::Transform tf_base_map_;
    tf::Transform tf_map_base_;
};
class VFHPathFollowAction
{
public:
    VFHPathFollowAction(std::string name);
    ~VFHPathFollowAction();
    void executeCB(const comn_pkg::PathFollowGoalConstPtr &goal);
    int getRange(float delta, float datas[]);
    void steer(Point& target, double &v, double &w);
private:
    ros::NodeHandle nh_;
    ros::ServiceServer service_;
    bool srvCallback(comn_pkg::VFHPlan::Request  &req,
                     comn_pkg::VFHPlan::Response &res );
    tf::TransformListener tfl_;
    ros::Publisher cmd_vel_pub_;
    std::string vel_topic_name_;
    ros::Subscriber laser_subscriber_;
    std::string laser_topic_;
    ros::Subscriber cmd_sub_;
    void laserCallback(const sensor_msgs::LaserScanConstPtr &laser_msg);
    actionlib::SimpleActionServer<comn_pkg::PathFollowAction> as_;
    comn_pkg::PathFollowFeedback feedback_;
    comn_pkg::PathFollowResult result_;
    std::string action_name_;
    std::string base_frame_, goal_frame_;
    double change_dist_;
    bool clearLaserBuff();
    bool subLaserTopic(std::string& topic_name);

    void cmdReceived(const char * cmd);
    void cmdCallback(const std_msgs::StringPtr cmd);
    bool is_back_;
    bool is_sub_;
    bool is_blind_;

    VFH m_vfhAvoider;
    LocalMap vfh_map_;
    LocalMap wait_map_;
    boost::recursive_mutex waitMapMutex;

    boost::recursive_mutex laserBufferMutex;
    RosLaserData *laser_buffer_;
    int curLaserBufferIndex;
    float laser_angle_min_;
    float laser_angle_max_;
    float laser_angle_increment_;
    int laser_num_;
    std::string laser_link_;
    tf::Stamped<tf::Pose> tf_laser_base_;
    bool has_tf_;
    const static int LaserBufferLen = 20;

    message_filters::Subscriber<sensor_msgs::LaserScan> *laser_sub;
    tf::MessageFilter<sensor_msgs::LaserScan> * laser_filter;

    ros::Publisher markers_publisher_;
    void publishObstacles(std::vector<Point2d> &obstacles);
    void publishArrows(std::vector<Point2d> &arrows);
    void publishPath(const std::vector<geometry_msgs::Point>& path, int cur);
    ros::Publisher map_publisher_;
    int publishMap(LocalMap& map);
    geometry_msgs::PoseStamped goalToBaseFrame(const geometry_msgs::PoseStamped& goal_pose_msg);

    double _param_a;
    double _param_b;
    double _param_c;
    double _max_v;
    double _min_v;
    double _max_rv;
    double _robot_radius;
    double _safe_dist;
    int _param_mu1;
    int _param_mu2;
    int _param_mu3;
    double _sector_angle;
    double _wide_angle;
    double _threshold_high;
    double _threshold_low;
    double win_sz;
    double m_delta;
    bool use_wait;
    double wait_x_;
    double wait_y_;
    double goal_tolerance_;
    int _clutter_const;
};
