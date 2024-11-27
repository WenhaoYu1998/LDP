#include "vfhlite.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/LaserScan.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <boost/thread.hpp>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
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
        tf_map_base_.setRotation(tf::Quaternion(0.707388,-0.706825, 0, 0));
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
class VFHNode
{
public:
    VFHNode(std::string name);
    ~VFHNode();
private:
    ros::NodeHandle nh_;
    tf::TransformListener tfl_;
    ros::Publisher cmd_vel_pub_;
    std::string vel_topic_name_;
    ros::Subscriber laser_subscriber_;
    std::string laser_topic_;
    ros::Subscriber goal_sub_;
    void laserCallback(const sensor_msgs::LaserScanConstPtr &laser_msg);
    void goalCallback(const geometry_msgs::PoseStamped& goal);
    geometry_msgs::PoseStamped goal_;
    bool has_goal_;
    std::string base_frame_;

    VFH m_vfhAvoider;
    LocalMap vfh_map_;
    tf::Stamped<tf::Pose> tf_laser_base_;
    bool has_tf_;

    ros::Publisher markers_publisher_;
    void publishObstacles(std::vector<Point2d> &obstacles);
    void publishArrows(std::vector<Point2d> &arrows);
    int publishMap(LocalMap& map);
    ros::Publisher map_publisher_;
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
    double goal_tolerance_;
    int _clutter_const;
};
