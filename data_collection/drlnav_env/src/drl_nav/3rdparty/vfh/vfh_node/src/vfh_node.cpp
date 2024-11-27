#include "vfh_node.h"

VFHNode::VFHNode(std::string name)
{
    ros::NodeHandle private_n("~");
    private_n.param("param_a", _param_a, 50.0);
    private_n.param("param_b", _param_b, 0.75);
    private_n.param("param_c", _param_c, 3.0);
    private_n.param("max_v", _max_v, 0.5);
    private_n.param("min_v", _min_v, 0.0);
    private_n.param("max_rv", _max_rv, 0.4);
    private_n.param("robot_radius", _robot_radius, 0.3);
    private_n.param("safe_dist", _safe_dist, 0.2);
    private_n.param("goal_tolerance", goal_tolerance_, 0.3);
    private_n.param("param_mu1", _param_mu1, 7);
    private_n.param("param_mu2", _param_mu2, 3);
    private_n.param("param_mu3", _param_mu3, 3);
    private_n.param("sector_angle", _sector_angle, 0.087);
    private_n.param("window_size", win_sz, 3.0);
    private_n.param("wide_angle", _wide_angle, 0.6);
    private_n.param("threshold_high", _threshold_high, 15000.);
    private_n.param("threshold_low", _threshold_low, 3500.);
    private_n.param("clutter_const", _clutter_const, 66000);
    private_n.param("base_fram_id", base_frame_, string("base_link"));
    private_n.param("vel_topic_name", vel_topic_name_, string("/cmd_vel"));
    private_n.param("laser_topic_name", laser_topic_, string("/scan"));
    private_n.param("delta", m_delta, 0.05);
    m_vfhAvoider.setVFHParam(_param_a, _param_b, _param_c, _max_v, _min_v, _max_rv,
                             _robot_radius, _safe_dist, _param_mu1, _param_mu2,
                             _param_mu3, _sector_angle, win_sz, _wide_angle,
                             _threshold_high, _threshold_low, _clutter_const);
    m_vfhAvoider.set_delta(m_delta);
    vfh_map_.init((int)ceil(win_sz / m_delta), (int)ceil(win_sz / m_delta), m_delta);
    has_tf_ = false;
    has_goal_ = false;
    laser_subscriber_ = nh_.subscribe(laser_topic_, 1, &VFHNode::laserCallback, this);
    goal_sub_ = nh_.subscribe("/vfh_goal", 1, &VFHNode::goalCallback, this);
    markers_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>("/vfh_marker_array", 100);
    map_publisher_ = nh_.advertise<nav_msgs::OccupancyGrid>("/vfh_map", 100);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(vel_topic_name_, 100);
}
VFHNode::~VFHNode()
{
}
geometry_msgs::PoseStamped VFHNode::goalToBaseFrame(const geometry_msgs::PoseStamped& goal_pose_msg)
{
  tf::Stamped<tf::Pose> goal_pose, base_pose;
  tf::poseStampedMsgToTF(goal_pose_msg, goal_pose);

  goal_pose.stamp_ = ros::Time();

  try{
    tfl_.transformPose(base_frame_, goal_pose, base_pose);
  }
  catch(tf::TransformException& ex){
    ROS_WARN("Failed to transform the goal pose from %s into the %s frame: %s",
        goal_pose.frame_id_.c_str(), base_frame_.c_str(), ex.what());
    return goal_pose_msg;
  }

  geometry_msgs::PoseStamped base_pose_msg;
  tf::poseStampedTFToMsg(base_pose, base_pose_msg);
  return base_pose_msg;
}

void VFHNode::goalCallback(const geometry_msgs::PoseStamped &goal)
{
    goal_ = goal;
    has_goal_ = true;
}

void VFHNode::laserCallback(const sensor_msgs::LaserScanConstPtr& laser_msg)
{
    if (has_tf_ == false)
    {
        tf::Stamped<tf::Pose> ident(tf::Transform(tf::createIdentityQuaternion(),
                                                  tf::Vector3(0, 0, 0)), ros::Time(), laser_msg->header.frame_id);
        try
        {
          tfl_.transformPose(base_frame_, ident, tf_laser_base_);
        }
        catch (tf::TransformException& e)
        {
          ROS_ERROR("Couldn't transform from %s to %s, "
          "even though the message notifier is in use", laser_msg->header.frame_id.c_str(), base_frame_.c_str());
          return;
        }
        std::cout << "laser_base_tf: " << tf_laser_base_.getOrigin().getX() << ","
                  << tf_laser_base_.getOrigin().getY() << ","
                  << tf_laser_base_.getOrigin().getZ() << std::endl;
        has_tf_ = true;
    }
    if (has_goal_ == false)
        return;
    geometry_msgs::PoseStamped goal_base = goalToBaseFrame(goal_);
    if (sqrt(goal_base.pose.position.x * goal_base.pose.position.x +
             goal_base.pose.position.y * goal_base.pose.position.y) < goal_tolerance_)
    {
        has_goal_ = false;
        geometry_msgs::Twist vel;
        cmd_vel_pub_.publish(vel);
        return;
    }
    const int RES = 1440 / 2;
    float delta = 6.28318f / RES;
    float lasdatas[2048];
    std::vector<Point2d> hits;
    hits.clear();
    vfh_map_.clear();
    int laser_num = laser_msg->ranges.size();
    double laser_angle_min = laser_msg->angle_min;
    double laser_angle_increment = laser_msg->angle_increment;
    for (int m = 0; m < laser_num; m++)
    {
        double theta = laser_angle_min + laser_angle_increment * m;
        double r = laser_msg->ranges[m];
        double x = r * cos(theta);
        double y = r * sin(theta);
        tf::Vector3 hit_laser;
        hit_laser.setValue(x, y, 0);
        tf::Vector3 hit_base = tf_laser_base_ * hit_laser;
        Point2d hit_base_p;
        hit_base_p.x = hit_base.x();
        hit_base_p.y = hit_base.y();
        hits.push_back(hit_base_p);
        int map_index = vfh_map_.getIndex(hit_base.x(), hit_base.y());
        if (map_index != -1)
        {
            vfh_map_.data_[map_index] = 1;
        }
    }
//    publishObstacles(hits);
//    publishMap(vfh_map_);
//    int res_val = 0;
    int idx = 0;
    std::vector<Point2d> arrows;
    for (float rot = 0; rot < 6.28318f; rot += delta, idx += 1)
    {
      float x, y, i = 0;
      float s = sin(rot);
      float c = cos(rot);
      while (1)
      {
        i = i + vfh_map_.delta_;
        x = i * c;
        y = i * s;
        int index_m = vfh_map_.getIndex(x, y);
        if (index_m == -1)
        {
          lasdatas[idx] = i;
          break;
        }
        else
        {
          if (vfh_map_.data_[index_m] == 1)
          {
            lasdatas[idx] = i;
            if (lasdatas[idx] < 0.01)
                continue;
            else
                break;
          }
        }
      }
      float arrow = lasdatas[idx];
      Point2d arrow_p(arrow*c, arrow*s);
      arrows.push_back(arrow_p);
    }
    publishObstacles(arrows);
    float *lasd = &lasdatas[0];
    OrientedPoint robot_pose(0,0,0);
    Point target(goal_base.pose.position.x, goal_base.pose.position.y);
    Point res = m_vfhAvoider.steer(robot_pose, target, lasd);
    double w = res.y;
    double v = res.x * cos(w);
    geometry_msgs::Twist vel;
    vel.angular.z = w;
    vel.linear.x = v;
    cmd_vel_pub_.publish(vel);
}

void VFHNode::publishArrows(std::vector<Point2d> &arrows)
{
    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.clear();
    int id = 0;
    visualization_msgs::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.color.a = 1;
    marker.color.r = 1;
    marker.color.g = 0;
    marker.color.b = 0;
    marker.action = visualization_msgs::Marker::ADD;
    marker.ns = "vfh_arrow";
    marker.type = visualization_msgs::Marker::ARROW;
    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    geometry_msgs::Point p;
    for (std::vector<Point2d>::iterator it = arrows.begin(); it != arrows.end(); it++)
    {
        marker.points.clear();
        p.x = 0;
        p.y = 0;
        p.z = 0;
        marker.points.push_back(p);
        p.x = it->x;
        p.y = it->y;
        marker.points.push_back(p);
        marker.id = id;
        marker_array.markers.push_back(marker);
        id ++;
    }
    markers_publisher_.publish(marker_array);
}
void VFHNode::publishObstacles(std::vector<Point2d> &obstacles)
{
    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.clear();
    int id = 0;
    visualization_msgs::Marker marker;
    marker.header.frame_id = base_frame_;
    marker.header.stamp = ros::Time::now();
    marker.color.a = 1;
    marker.color.r = 0;
    marker.color.g = 0;
    marker.color.b = 1;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.z = 0;
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.ns = "vfh_obstacle";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    for (std::vector<Point2d>::iterator it = obstacles.begin(); it != obstacles.end(); it++)
    {

        marker.pose.position.x = it->x;
        marker.pose.position.y = it->y;
        marker.id = id;
        marker_array.markers.push_back(marker);
        id ++;
    }
    markers_publisher_.publish(marker_array);
}

int VFHNode::publishMap(LocalMap &map)
{
    nav_msgs::OccupancyGrid map_msg;
    map_msg.header.frame_id = base_frame_;
    map_msg.header.stamp = ros::Time::now();
    map_msg.info.height = map.size_x_;
    map_msg.info.width = map.size_y_;
    map_msg.info.origin.position.x = map.size_x_ * map.delta_ / 2.0;
    map_msg.info.origin.position.y = map.size_y_ * map.delta_ / 2.0;
    map_msg.info.origin.orientation.w = -3.27724e-05;
    map_msg.info.origin.orientation.x = 0.707388;
    map_msg.info.origin.orientation.y = -0.706825;
    map_msg.info.origin.orientation.z = 3.27463e-05;
    map_msg.info.resolution = map.delta_;
    map_msg.data.clear();
    int j = 0;
    for (int i = 0; i < map.size_x_ * map.size_y_; i++)
    {
        if (map.data_[i] == 0)
            map_msg.data.push_back(-1);
        else
        {
            j++;
            map_msg.data.push_back(100);
        }
    }
    map_publisher_.publish(map_msg);
    return j;
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "vfh_node");
    VFHNode my_vfh("vfh_node");
    ros::spin();
    return 0;
}
