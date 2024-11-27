#include "vfh_path_follower.h"

VFHPathFollowAction::VFHPathFollowAction(std::string name) :
    as_(nh_, name, boost::bind(&VFHPathFollowAction::executeCB, this, _1), false),
    action_name_(name)
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
    private_n.param("wait_x", wait_x_, 2.0);
    private_n.param("wait_y", wait_y_, 0.6);
    private_n.param("use_wait", use_wait, true);
    private_n.param("base_fram_id", base_frame_, string("base_link"));
    private_n.param("goal_fram_id", goal_frame_, string("odom"));
    private_n.param("vel_topic_name", vel_topic_name_, string("/cmd_vel"));
    private_n.param("change_dist", change_dist_, 0.2);
    private_n.param("delta", m_delta, 0.05);

    m_vfhAvoider.setVFHParam(_param_a, _param_b, _param_c, _max_v, _min_v, _max_rv,
                             _robot_radius, _safe_dist, _param_mu1, _param_mu2,
                             _param_mu3, _sector_angle, win_sz, _wide_angle,
                             _threshold_high, _threshold_low, _clutter_const);
    m_vfhAvoider.set_delta(m_delta);
    vfh_map_.init((int)ceil(win_sz / m_delta), (int)ceil(win_sz / m_delta), m_delta);
    wait_map_.init((int)ceil(wait_x_ / m_delta), (int)ceil(wait_y_ / m_delta) , m_delta);
    has_tf_ = false;
    is_back_ = false;
    is_sub_ = false;
    is_blind_ = false;
    laser_topic_ = "non_name";

    laser_buffer_ = new RosLaserData[LaserBufferLen];
    curLaserBufferIndex = -1;

    cmd_sub_ = nh_.subscribe("/path_follower_cmd", 1000, &VFHPathFollowAction::cmdCallback, this);
    markers_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>("/vfh_marker_array", 100);
    map_publisher_ = nh_.advertise<nav_msgs::OccupancyGrid>("/vfh_map", 100);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(vel_topic_name_, 100);
    service_ = private_n.advertiseService("vfh_plan", &VFHPathFollowAction::srvCallback, this);
    ROS_INFO("Start plan service.");

    as_.start();
    ROS_INFO("Start action service.");
}
VFHPathFollowAction::~VFHPathFollowAction()
{
    clearLaserBuff();
    delete[] laser_buffer_;
    if (is_sub_)
    {
        delete laser_sub;
        delete laser_filter;
        laser_sub = NULL;
        laser_filter = NULL;
        is_sub_ = false;
    }
}

geometry_msgs::PoseStamped VFHPathFollowAction::goalToBaseFrame(const geometry_msgs::PoseStamped& goal_pose_msg)
{
  tf::Stamped<tf::Pose> goal_pose, base_pose;
  tf::poseStampedMsgToTF(goal_pose_msg, goal_pose);

  goal_pose.stamp_ = goal_pose_msg.header.stamp;

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

bool VFHPathFollowAction::srvCallback(comn_pkg::VFHPlan::Request  &req,
                                 comn_pkg::VFHPlan::Response &res )
{
    if (has_tf_ == false)
    {
        tf::Stamped<tf::Pose> ident(tf::Transform(tf::createIdentityQuaternion(),
                                                  tf::Vector3(0, 0, 0)), ros::Time(), req.scan.header.frame_id);
        try
        {
          tfl_.transformPose(base_frame_, ident, tf_laser_base_);
        }
        catch (tf::TransformException& e)
        {
          ROS_ERROR("Couldn't transform from %s to %s, "
          "even though the message notifier is in use", req.scan.header.frame_id.c_str(), base_frame_.c_str());
          return false;
        }
        std::cout << "laser_base_tf: " << tf_laser_base_.getOrigin().getX() << ","
                  << tf_laser_base_.getOrigin().getY() << ","
                  << tf_laser_base_.getOrigin().getZ() << std::endl;
        has_tf_ = true;
    }
    int laser_num = req.scan.ranges.size();
    double laser_angle_min = req.scan.angle_min;
    double laser_angle_increment = req.scan.angle_increment;
    vfh_map_.clear();
    for (int m = 0; m < laser_num; m++)
    {
        double theta = laser_angle_min + laser_angle_increment * m;
        double r = req.scan.ranges[m];
        double x = r * cos(theta);
        double y = r * sin(theta);
        tf::Vector3 hit_laser;
        hit_laser.setValue(x, y, 0);
        tf::Vector3 hit_base = tf_laser_base_ * hit_laser;
        int map_index = vfh_map_.getIndex(hit_base.x(), hit_base.y());
        if (map_index != -1)
        {
            vfh_map_.data_[map_index] = 1;
        }
    }

    int idx = 0;
    std::vector<Point2d> arrows;
    const int RES = 1440 / 2;
    float delta = 6.28318f / RES;
    float datas[2048];
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
          datas[idx] = i;
          break;
        }
        else
        {
          if (vfh_map_.data_[index_m] == 1)
          {
            datas[idx] = i;
            if (datas[idx] < 0.01)
                continue;
            else
                break;
          }
        }
      }
      float arrow = datas[idx];
      Point2d arrow_p(arrow*c, arrow*s);
      arrows.push_back(arrow_p);
    }
    publishObstacles(arrows);

    float *lasd = &datas[0];
    OrientedPoint robot_pose(0,0,0);
    m_vfhAvoider.set_delta(0.05);
    geometry_msgs::PoseStamped target_base = goalToBaseFrame(req.target);
    Point target(target_base.pose.position.x, target_base.pose.position.y);
    Point vw = m_vfhAvoider.steer(robot_pose, target, lasd);
    res.vel.angular.z = vw.y;
    res.vel.linear.x = vw.x * cos(res.vel.angular.z);

//    if (vw.y >= 0)
//        res.vel.angular.z = std::min(vw.y, _max_rv);
//    else
//        res.vel.angular.z = std::max(vw.y, -_max_rv);
//    res.vel.linear.x = vw.x * cos(res.vel.angular.z);
    return true;
}

bool VFHPathFollowAction::clearLaserBuff()
{
    laserBufferMutex.lock();
    for (int i = 0; i < LaserBufferLen; i++)
    {
      if(laser_buffer_[i].data != NULL)
      {
          delete[] laser_buffer_[i].data;
          laser_buffer_[i].data = NULL;
      }
    }
    curLaserBufferIndex = -1;
    has_tf_ = false;
    laserBufferMutex.unlock();
    return true;
}
bool VFHPathFollowAction::subLaserTopic(string &topic_name)
{
    if (is_sub_)
    {
        laser_sub->unsubscribe();
        delete laser_filter;
        delete laser_sub;
        laser_sub = NULL;
        laser_filter = NULL;
        clearLaserBuff();
    }
    laser_topic_ = topic_name;
    laser_sub = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, laser_topic_, 40);
    laser_filter = new tf::MessageFilter<sensor_msgs::LaserScan>(*laser_sub, tfl_, goal_frame_, 40);
    laser_filter->registerCallback(boost::bind(&VFHPathFollowAction::laserCallback, this, _1));
    is_sub_ = true;
}
void VFHPathFollowAction::cmdReceived(const char * cmd)
{
    ROS_INFO("enter cmd");
    float f1, f2, f3, f4, f5;
    char str1[20];
    int d1;
    if (PEEK_CMD_S(cmd, "backward", 8, str1))
    {
        is_back_ = true;
        is_blind_ = false;
        std::string topic_name = str1;
        if (topic_name != laser_topic_)
            subLaserTopic(topic_name);
        std::cout << "backward move" << std::endl;
    }
    else if (PEEK_CMD_S(cmd, "forward", 7, str1))
    {
        is_back_ = false;
        is_blind_ = false;
        std::string topic_name = str1;
        subLaserTopic(topic_name);
        ROS_INFO("forward");
        std::cout << "forward move" << std::endl;
    }
    else if(PEEK_CMD(cmd, "reset"))
    {
        is_back_ = false;
        is_blind_ = false;
        if (is_sub_)
        {
            laser_sub->unsubscribe();
            delete laser_filter;
            delete laser_sub;
            laser_sub = NULL;
            laser_filter = NULL;
            clearLaserBuff();
        }
        is_sub_ = false;
        laser_topic_ = "non_name";
        std::cout << "reset" << std::endl;
    }
    else if(PEEK_CMD_D(cmd, "wait", 4, d1))
    {
        if (d1 == 0)
            use_wait = false;
        else
            use_wait = true;
        std::cout << "wait: " << use_wait << std::endl;
    }
    else if(PEEK_CMD_D(cmd, "blind", 5, d1))
    {
        if (d1 == 0)
            is_blind_ = false;
        else
            is_blind_ = true;
       std::cout << "blind: " << use_wait << std::endl;
    }
    else if(PEEK_CMD_FF(cmd, "wait_horizon", 12, f1, f2))
    {
        waitMapMutex.lock();
        wait_map_.free();
        wait_x_ = f1;
        wait_y_ = f2;
        wait_map_.init((int)ceil(wait_x_ / m_delta), (int)ceil(wait_y_ / m_delta) , m_delta);
        waitMapMutex.unlock();
        std::cout << "wait_horizon: " << f1 << ", " << f2 << std::endl;
    }
    ROS_INFO("leave cmd");
}
void VFHPathFollowAction::cmdCallback(const std_msgs::StringPtr cmd)
{
    ROS_INFO("Received cmd: %s", cmd->data.c_str());
    cmdReceived(cmd->data.c_str());
}
void VFHPathFollowAction::laserCallback(const sensor_msgs::LaserScanConstPtr& laser_msg)
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
    laserBufferMutex.lock();
    laser_num_ = laser_msg->ranges.size();
    laser_angle_increment_ = laser_msg->angle_increment;
    laser_angle_min_ = laser_msg->angle_min;
    laser_angle_max_ = laser_msg->angle_max;
    laser_link_ = laser_msg->header.frame_id;
    curLaserBufferIndex = (curLaserBufferIndex + 1) % LaserBufferLen;
    //ROS_INFO("%d", curLaserBufferIndex);
    RosLaserData & laserData = laser_buffer_[curLaserBufferIndex];
    if (laserData.data == NULL)
      laserData.data = new double[laser_num_];
    laserData.stamp = laser_msg->header.stamp;
    for (int i = 0; i < laser_num_; i++)
        laserData.data[i] = laser_msg->ranges[i];
    laserBufferMutex.unlock();
}
int VFHPathFollowAction::getRange(float delta, float datas[])
{
    std::vector<Point2d> hits;
    hits.clear();
    vfh_map_.clear();
    waitMapMutex.lock();
    wait_map_.clear();
    if ((curLaserBufferIndex != -1) && (is_blind_ == false))
    {
        RosLaserData& laser_data = laser_buffer_[curLaserBufferIndex];
        for (int m = 0; m < laser_num_; m++)
        {
            double theta = laser_angle_min_ + laser_angle_increment_ * m;
            double r = laser_data.data[m];
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
            int wait_map_index = wait_map_.getIndex(hit_base.x(), hit_base.y());
            if (wait_map_index != -1 && fabs(hit_base.x()) > 0.05 && fabs(hit_base.y()) > 0.05)
            {
                wait_map_.data_[wait_map_index] = 1;
            }
        }
    }
//    publishObstacles(hits);
//    publishMap(vfh_map_);
//    int res_val = 0;
    int res_val = publishMap(wait_map_);
    waitMapMutex.unlock();
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
          datas[idx] = i;
          break;
        }
        else
        {
          if (vfh_map_.data_[index_m] == 1)
          {
            datas[idx] = i;
            if (datas[idx] < 0.01)
                continue;
            else
                break;
          }
        }
      }
      float arrow = datas[idx];
      Point2d arrow_p(arrow*c, arrow*s);
      arrows.push_back(arrow_p);
    }
    publishObstacles(arrows);
    return res_val;
}

void VFHPathFollowAction::steer(Point &target, double& v, double& w)
{
    const int RES = 1440 / 2;
    float delta = 6.28318f / RES;
    float lasdatas[2048];
    int detect_obj;
    laserBufferMutex.lock();
    detect_obj = getRange(delta, lasdatas);
    laserBufferMutex.unlock();
    if (detect_obj >= 2 && use_wait == true)
    {
        std::cout << "obstacles count: " << detect_obj <<std::endl;
        ROS_INFO("wait");
        feedback_.status.data = "wait";
        v = 0;
        w = 0;
        return;
    }
    float *lasd = &lasdatas[0];
    OrientedPoint robot_pose(0,0,0);
    if (is_back_)
        robot_pose.theta = _PI;
    m_vfhAvoider.set_delta(0.05);
    Point res = m_vfhAvoider.steer(robot_pose, target, lasd);
    w = res.y;
    v = res.x * cos(w);
    if (is_back_)
        v = - fabs(v);
}

void VFHPathFollowAction::executeCB(const comn_pkg::PathFollowGoalConstPtr &goal)
{
    ros::Rate r(30);
    std::cout << "recieve goal" << std::endl;
    bool success = true;
    bool goal_new = false;
    int i = 0;
    std::vector<geometry_msgs::Point> path = goal->path;
    while (i < path.size())
    {
        publishPath(path, i);
        feedback_.id = i;
        feedback_.status.data = "normal";
        bool is_move = true;
        while(is_move)
        {
            if (as_.isPreemptRequested())
            {
                if(as_.isNewGoalAvailable())
                {
                    comn_pkg::PathFollowGoal new_goal =  *as_.acceptNewGoal();
                    path = new_goal.path;
                    i = 0;
                    goal_new = true;
                }
                else
                {
                    ROS_INFO("%s: Preempted", action_name_.c_str());
                    as_.setPreempted();
                    success = false;
                    geometry_msgs::Twist vel_msg;
                    vel_msg.linear.x = 0;
                    vel_msg.angular.z = 0;
                    feedback_.v = 0;
                    feedback_.w = 0;
                    feedback_.status.data = "cancel";
                    as_.publishFeedback(feedback_);
                    std::cout << "preempt! publish 0 0" <<std::endl;
                    cmd_vel_pub_.publish(vel_msg);
                    return;
                }
            }
            if (is_sub_ == false)
            {
                ROS_WARN("not subscribe any laser topic, blind move!");
                is_blind_ = true;
            }
            tf::Vector3 cur_goal;
            cur_goal.setValue(path[i].x, path[i].y, 0);
            tf::StampedTransform m_tf;
            ros::Time cur = ros::Time::now();
            tfl_.waitForTransform(base_frame_, goal_frame_, cur, ros::Duration(3.0));
            try {
              tfl_.lookupTransform(base_frame_, goal_frame_, cur, m_tf);
            } catch(tf::TransformException& ex){
              ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
              return;
            }
            tf::Vector3 cur_goal_base = m_tf * cur_goal;
            double dist = sqrt(cur_goal_base.x() * cur_goal_base.x() +
                    cur_goal_base.y() * cur_goal_base.y());
            feedback_.distance = dist;
            Point target(cur_goal_base.x(), cur_goal_base.y());
            if (i == (path.size()-1))
            {
                if (dist < goal_tolerance_)
                {
                    geometry_msgs::Twist vel_msg;
                    vel_msg.linear.x = 0;
                    vel_msg.angular.z = 0;
                    feedback_.v = 0;
                    feedback_.w = 0;
                    feedback_.status.data = "arrival";
                    std::cout << "publish 0 0" <<std::endl;
                    cmd_vel_pub_.publish(vel_msg);
                    is_move = false;
                    break;
                }
                else
                {
                    double v, w;
                    steer(target, v, w);
                    geometry_msgs::Twist vel_msg;
                    vel_msg.linear.x = v;
                    vel_msg.angular.z = w;
                    feedback_.v = v;
                    feedback_.w = w;
                    cmd_vel_pub_.publish(vel_msg);
                }

            }
            if (i != (path.size()-1))
            {
                if (dist < change_dist_)
                {
                    is_move = false;
                    feedback_.v = 0;
                    feedback_.w = 0;
                    as_.publishFeedback(feedback_);
                    break;
                }
                else
                {
                    double v, w;
                    steer(target, v, w);
                    geometry_msgs::Twist vel_msg;
                    vel_msg.linear.x = v;
                    vel_msg.angular.z = w;
                    feedback_.v = v;
                    feedback_.w = w;
                    cmd_vel_pub_.publish(vel_msg);
                }
            }
            as_.publishFeedback(feedback_);
            r.sleep();
        }
        if (goal_new == false)
            i += 1;

    }
    if(success)
    {
      result_.result = 0;
      ROS_INFO("%s: Succeeded", action_name_.c_str());
      as_.setSucceeded(result_);
    }
}

void VFHPathFollowAction::publishArrows(std::vector<Point2d> &arrows)
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
void VFHPathFollowAction::publishObstacles(std::vector<Point2d> &obstacles)
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

int VFHPathFollowAction::publishMap(LocalMap &map)
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

void VFHPathFollowAction::publishPath(const std::vector<geometry_msgs::Point> &path, int cur)
{
    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker arrow;
    arrow.header.stamp = ros::Time(0);
    arrow.header.frame_id = goal_frame_;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.color.a = 1;
    arrow.color.r = 0;
    arrow.color.g = 0;
    arrow.color.b = 1;
    arrow.ns = "vfh_path_arrow";
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.scale.x = 0.02;
    arrow.scale.y = 0.02;
    visualization_msgs::Marker node;
    node.header.stamp = ros::Time(0);
    node.header.frame_id = goal_frame_;
    node.action = visualization_msgs::Marker::ADD;
    node.color.a = 1;
    node.color.r = 0;
    node.color.g = 0;
    node.color.b = 1;
    node.ns = "vfh_path_point";
    node.type = visualization_msgs::Marker::SPHERE;
    node.scale.x = 0.1;
    node.scale.y = 0.1;
    node.scale.z = 0.1;
    node.pose.orientation.w = 1;
    int id = 0;
    geometry_msgs::Point p;
    for (int i = 0; i < int(path.size()-1); i++)
    {
        arrow.id = id;
        arrow.points.clear();
        p.x = path[i].x;
        p.y = path[i].y;
        arrow.points.push_back(p);
        p.x = path[i+1].x;
        p.y = path[i+1].y;
        arrow.points.push_back(p);

        node.id = id;
        node.pose.position.x = path[i].x;
        node.pose.position.y = path[i].y;
        if (cur == i+1)
        {
            arrow.color.r = 1;
            arrow.color.g = 0;
            arrow.color.b = 0;
        }
        if (cur == i)
        {
            node.color.r = 1;
            node.color.g = 0;
            node.color.b = 0;
        }
        id++;
        markers.markers.push_back(arrow);
        markers.markers.push_back(node);
    }
    node.id = id + 1;
    node.pose.position.x = path[path.size() -1].x;
    node.pose.position.y = path[path.size() -1].y;
    if (cur == int(path.size() -1))
    {
        node.color.r = 1;
        node.color.g = 0;
        node.color.b = 0;
    }
    markers.markers.push_back(node);
    markers_publisher_.publish(markers);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vfh_planner");
    VFHPathFollowAction my_action("vfh_planner");
    ros::spin();
    return 0;
}
