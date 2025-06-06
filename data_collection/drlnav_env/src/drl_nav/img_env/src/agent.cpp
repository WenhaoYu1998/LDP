#include "agent.h"
#include <math.h>
#include <algorithm>
#include <random>
#include <ctime>


Agent::Agent(string ktype)
{
    ktype_ = ktype;
}

Agent::Agent(string ktype, double step_hz)
{
    ktype_ = ktype;
    step_hz_ = step_hz;
}


void Agent::init_shape_circle(const vector<double> &sizes, vector<Point2d>& bbox){
    double resolution = 0.01;
    int bb = ceil(sizes[2] / resolution);
    for (int m = -bb; m <= bb; m++)
        for (int n = -bb; n <= bb; n++)
        {
            if (sqrt(m * resolution * m * resolution + n * resolution * n* resolution) <= sizes[2])
            {
                Point2d p(m*resolution + sizes[0], n*resolution + sizes[1]);
                bbox.push_back(p);
            }
        }
}

// beep circle 就外面一圈
void Agent::init_shape_beep(const vector<double> &sizes, double beep_r){
    double resolution = 0.01;
    int bb = ceil(beep_r / resolution);
    // r - 0.01
    double x  = beep_r - 0.01;
    for (int m = -bb; m <= bb; m++)
        for (int n = -bb; n <= bb; n++)
        {
            double d = sqrt(m * resolution * m * resolution + n * resolution * n* resolution);
            if (d <= beep_r && d >= x)
            {
                Point2d p(m*resolution + sizes[0], n*resolution + sizes[1]);
                bbox_beep_.push_back(p);
            }
        }
}


void Agent::init_shape_rectangle(const vector<double> &sizes, vector<Point2d>& bbox){
    double resolution = 0.01;
    l_f = (sizes[1]);
    l_r = (sizes[1]);
    int x_min = floor(sizes[0] / resolution);
    int x_max = ceil(sizes[1] / resolution);
    int y_min = floor(sizes[2] / resolution);
    int y_max = ceil(sizes[3] / resolution);
    for (int m = x_min; m <= x_max; m++)
        for (int n = y_min; n <= y_max; n++) {
            Point2d p(m * resolution, n * resolution);
            bbox.push_back(p);
        }
}

void Agent::init_shape(string shape, const vector<double> &sizes)
{
    bbox_.clear();
    if (shape == "circle")
    {
        init_shape_circle(sizes, bbox_);
    }
    else if (shape == "rectangle") {
        init_shape_rectangle(sizes, bbox_);
    }

    shape_ = shape;
    sizes_ = sizes;
}

void Agent::init_view_map(double width, double height, double resolution)
{
    view_map_.resolution_ = resolution;
    view_map_.img_width_ = width / resolution;
    view_map_.img_height_ = height / resolution;
    tf_view_base_.setOrigin(tf::Vector3(height/2, width/2, 0));
    tf::Quaternion q;
    q.setRPY(0, 0, 3.14159);
    tf_view_base_.setRotation(q);
    tf_base_view_ = tf_view_base_.inverse();
    control_hz_ = 0.05;
}

void Agent::base2view(const Point2d &xy_base, Point2d &xy_view)
{
    tf::Vector3 xyv_base(xy_base.x, xy_base.y, 0);
    tf::Vector3 xyv_view = tf_base_view_ * xyv_base;
    xy_view.x = xyv_view.getX();
    xy_view.y = xyv_view.getY();
}

void Agent::view2base(const Point2d &xy_view, Point2d &xy_base)
{
    tf::Vector3 xyv_view(xy_view.x, xy_view.y, 0);
    tf::Vector3 xyv_base = tf_view_base_ * xyv_view;
    xy_base.x = xyv_base.getX();
    xy_base.y = xyv_base.getY();
}

bool Agent::base2map(GridMap &grid_map, Vec3b value, string target_frame, const Point2d& base, Point2i& map)
{
    tf::Transform tf_base_world = get_base_world();
    if (target_frame == "world_map")
    {
        Point2d bb_world;
        base2world(base, bb_world, tf_base_world);
        grid_map.world2map(bb_world, map);
    }
    else if (target_frame == "map")
        grid_map.world2map(base, map);
    else if (target_frame == "view_map")
    {
        Point2d bb_world;
        base2view(base, bb_world);
        grid_map.world2map(bb_world, map);
    }
    return true;
}

void Agent::base2world(const Point2d &xy_base, Point2d &xy_world, tf::Transform tf_base_world)
{
//    if (tf_base_world == NULL)
//    tf_base_world = get_base_world();
    tf::Vector3 xyv_base(xy_base.x, xy_base.y, 0);
    tf::Vector3 xyv_world = tf_base_world * xyv_base;
    xy_world.x = xyv_world.getX();
    xy_world.y = xyv_world.getY();
}

tf::Transform Agent::get_base_world()
{
    tf::Transform tf_base_world;
    tf_base_world.setOrigin(tf::Vector3(robot_pose_.x, robot_pose_.y, 0));
    tf::Quaternion q;
    q.setRPY(0, 0, robot_pose_.z);
    tf_base_world.setRotation(q);
    return tf_base_world;
}

tf::Transform Agent::get_view_world()
{
    return get_base_world() * tf_view_base_;
}

void Agent::init_pose(const Point3d &pose)
{
    robot_pose_ = pose;
    odom_pose_ = Point3d(0, 0, 0);
    last_vw_ = Point2d(0, 0);
    tf_odom_world_.setOrigin(tf::Vector3(pose.x, pose.y, 0));
    tf::Quaternion q;
    q.setRPY(0, 0, pose.z);
    tf_odom_world_.setRotation(q);
}

void Agent::set_goal(const Point3d &pose)
{
    target_pose_.x = pose.x;
    target_pose_.y = pose.y;
    target_pose_.z = robot_pose_.z;
    tf_target_world_.setOrigin(tf::Vector3(target_pose_.x, target_pose_.y, 0));
    tf::Quaternion q;
    q.setRPY(0, 0, target_pose_.z);
    tf_target_world_.setRotation(q);
    tf_world_target_ = tf_target_world_.inverse();
}

bool Agent::draw_line(GridMap &grid_map, Vec3b value, string target_frame, vector<Point2i>& bbox)
{
    for (int i = 0; i < bbox.size() - 1; i++) {
        /* Opnecv line 绘制的坐标系和 Mat不同 因此这里使用 Mat.at进行轨迹绘制 */
        // line(grid_map.map_, bbox[i], bbox[i + 1], value, 2);
        if (grid_map.is_in_map(bbox[i]))
        {
            grid_map.map_.at<Vec3b>(bbox[i].x, bbox[i].y) = value;
        }
    }
    return true;
}

void Agent::get_state(vector<double> &state)
{
    state.clear();
    tf::Transform tf_target_base = (tf_world_target_ * get_base_world()).inverse();
    tf::Vector3 t = tf_target_base.getOrigin();
    state.push_back(t.x());
    state.push_back(t.y());
    
    if (state_dim_ == 3)
    {
        tf::Quaternion q = tf_target_base.getRotation();
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        state.push_back(yaw);
    }
    else if (state_dim_ == 4)
    {
        state.push_back(last_vw_.x);
        state.push_back(last_vw_.y);
    }
}

bool Agent::cmd(double a, double w)
{
    bool is_arrive = false;
    last_vw_.x = a;
    last_vw_.y = w;

    // 使用加速度和前轮转角控制
    if(ktype_ == "carlike_aw")
    {
        odom_pose_ = robot_pose_;
        double cur_control = 0;
        // double cur_steering_angle = 0; // 当前前轮转角
        // 计算加速度
        double acceleration = a;
        double cur_steering_angle = w;
        while (cur_control < step_hz_)
        {
            // 更新机器人位置
            //if(cur_velocity_ > 0.59)
            //{
            //    odom_pose_.x += cur_velocity_ * control_hz_ * cos(odom_pose_.z);
            //    odom_pose_.y += cur_velocity_ * control_hz_ * sin(odom_pose_.z);
            //    odom_pose_.z += (tan(cur_steering_angle) / (l_r + l_f)) * cur_velocity_ * control_hz_;
            //}
            //else{
                odom_pose_.x += ((cur_velocity_ * control_hz_) + 1/2 * acceleration * pow(control_hz_, 2)) * cos(odom_pose_.z);
                odom_pose_.y += ((cur_velocity_ * control_hz_) + 1/2 * acceleration * pow(control_hz_, 2)) * sin(odom_pose_.z);
                odom_pose_.z += (tan(cur_steering_angle) / (l_r + l_f)) * ((cur_velocity_ * control_hz_) + 1/2 * acceleration * pow(control_hz_, 2));
            //}
            
            run_dis_episode_ += sqrt(pow(((cur_velocity_ * control_hz_) + 1/2 * acceleration * pow(control_hz_, 2)) * cos(odom_pose_.z), 2) + pow(((cur_velocity_ * control_hz_) + 1/2 * acceleration * pow(control_hz_, 2)) * sin(odom_pose_.z), 2));
            
            // 更新速度和前轮转角
            cur_velocity_ += acceleration * control_hz_;
            
            if(cur_velocity_ > 0.6)
            	cur_velocity_ = 0.6;
            if(cur_velocity_ < -0.6)
            	cur_velocity_ = -0.6;
            	
            // cur_steering_angle = w;

            // 计算当前机器人到目标位置的距离
            double cur_dist = sqrt((odom_pose_.x - target_pose_.x) * (odom_pose_.x - target_pose_.x) + (odom_pose_.y - target_pose_.y) * (odom_pose_.y - target_pose_.y));
            refresh_num_episode_ += 1;
            
            if (cur_dist <= 0.3)
            {
                is_arrive = true;
                break;
            }
            
            cur_control += control_hz_;
        }
        double theta = robot_pose_.z;
        double dt = step_hz_;
        // run_dis_episode_ += sqrt(pow(v * dt * cos(theta), 2) + pow(v * dt * sin(theta), 2));
        // robot_pose_.x += cur_velocity * dt * cos(theta);
        // robot_pose_.y += cur_velocity * dt * sin(theta);
        // robot_pose_.z += (tan(cur_steering_angle) / (l_r + l_f)) * cur_velocity * dt;
        robot_pose_.x = odom_pose_.x;
        robot_pose_.y = odom_pose_.y;
        robot_pose_.z = odom_pose_.z;
    }
    else if(ktype_ == "carlike"){
        odom_pose_ = robot_pose_;
        double cur_control = 0;
        cur_velocity_ = a;
        while(cur_control <= step_hz_){
            odom_pose_.x += a * control_hz_ * cos(odom_pose_.z);
            odom_pose_.y += a * control_hz_ * sin(odom_pose_.z);
            odom_pose_.z += (tan(w)/(l_r + l_f)) * a * control_hz_;

            run_dis_episode_ += sqrt(pow(a * control_hz_ * cos(odom_pose_.z), 2) + pow(a * control_hz_ * sin(odom_pose_.z), 2));

            double cur_dist = sqrt((robot_pose_.x - target_pose_.x) * (robot_pose_.x - target_pose_.x)
                                   + (robot_pose_.y - target_pose_.y) * (robot_pose_.y - target_pose_.y));
            if (cur_dist <= 0.3)
            {
                is_arrive = true;
                break;
            }
            cur_control += control_hz_;
            refresh_num_episode_ += 1;
        }
        double theta = robot_pose_.z;
        double dt = step_hz_;
        robot_pose_.x += a * dt * cos(theta);
        robot_pose_.y += a * dt * sin(theta);
        robot_pose_.z += (tan(w)/(l_r + l_f)) * a * dt;
        if (a < 0)
        {
            robot_back_state = 1;
        }
        else
        {
            robot_back_state = 0;
        }
        
    }

    double cur_dist = sqrt((robot_pose_.x - target_pose_.x) * (robot_pose_.x - target_pose_.x)
                                   + (robot_pose_.y - target_pose_.y) * (robot_pose_.y - target_pose_.y));
    if (cur_dist <= 0.3)
    {
        is_arrive = true;
    }
    is_arrive_ = is_arrive;
    return is_arrive;
}

void Agent::trajectory_visual(vector<float> v, vector<float> w, GridMap &grid_map, Vec3b value, string target_frame){
    double points_hz = 0.01;
    Point3d pose = robot_pose_;
    traj_points_.clear();
    visual_traj_.clear();

    // 将初始点加入traj数组
    Point2d temp_point;
    temp_point.x = pose.x;
    temp_point.y = pose.y;
    traj_points_.push_back(temp_point);

    for (int i = 0; i < v.size(); i++)
    {
        double cur_control = 0;
        while (cur_control <= step_hz_)
        {
            pose.x += v[i] * points_hz * cos(pose.z);
            pose.y += v[i] * points_hz * sin(pose.z);
            pose.z += (tan(w[i])/(l_r + l_f)) * v[i] * points_hz;
            temp_point.x = pose.x;
            temp_point.y = pose.y;
            traj_points_.push_back(temp_point);

            cur_control += points_hz;
        }
    }

    for (int i = 0; i < traj_points_.size(); i++)
    {
        tf::Vector3 xy_world(traj_points_[i].x, traj_points_[i].y, 0);
        tf::Vector3 xy_base = get_base_world().inverse() * xy_world;
        Point2d traj_point;
        Point2i visual_point;
        traj_point.x = xy_base.getX();
        traj_point.y = xy_base.getY();
        base2map(grid_map, value, target_frame, traj_point, visual_point);
        visual_traj_.push_back(visual_point);
    }
    
}

void Agent::global_path_visual(vector<int> global_path_x, vector<int> global_path_y, GridMap &grid_map, Vec3b value, string target_frame){
    // global_path_points_.clear();
    visual_global_path_.clear();
    // for (int i = 0; i < global_path_x.size(); i++)
    // {
    //     Point2d temp_point;
    //     temp_point.x = global_path_x[i];
    //     temp_point.y = global_path_y[i];
    //     global_path_points_.push_back(temp_point);
    // }

    // for (int i = 0; i < global_path_points_.size(); i++)
    // {
    //     tf::Vector3 xy_world(global_path_points_[i].x, global_path_points_[i].y, 0);
    //     tf::Vector3 xy_base = get_base_world().inverse() * xy_world;
    //     Point2d traj_point;
    //     Point2i visual_point;
    //     traj_point.x = xy_base.getX();
    //     traj_point.y = xy_base.getY();
    //     base2map(grid_map, value, target_frame, traj_point, visual_point);
    //     visual_global_path_.push_back(visual_point);
    // }
    for (int i = 0; i < global_path_x.size(); i++){
        Point2i visual_point;
        visual_point.x = global_path_x[i];
        visual_point.y = global_path_y[i];
        visual_global_path_.push_back(visual_point);
    }
}

int Agent::draw(GridMap &grid_map, int value, string target_frame, vector<Point2d>& bbox)
/*
    返回值：
        0表示没撞
        1撞障碍物
        2撞行人
        3撞别的小车
*/
{
    int is_collision = 0;
    tf::Transform tf_base_world = get_base_world();
    for (vector<Point2d>::iterator it = bbox.begin(); it != bbox.end(); it++)
    {
        Point2i bb_m;
        if (target_frame == "world_map")
        {
            Point2d bb_world;
            base2world(*it, bb_world, tf_base_world);
            grid_map.world2map(bb_world, bb_m);
        }
        else if (target_frame == "map")
            grid_map.world2map(*it, bb_m);
        else if (target_frame == "view_map")
        {
            Point2d bb_world;
            base2view(*it, bb_world);
            grid_map.world2map(bb_world, bb_m);
        }
        if (grid_map.is_in_map(bb_m))
        {
            if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 0) //障碍物标记符为0
                is_collision = 1;  // 撞了障碍物
            else if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 1) // 行人标记符为1
                is_collision = 2;  // 撞了行人
            else if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 2) // 小车标记符为2
                is_collision = 3;  // 撞了别的小车
            else if (value >= 0)
                grid_map.map_.at<uchar>(bb_m.x, bb_m.y) = value; // 往占位地图里填充占位标志。
        }
    }

    return is_collision;
}

void Agent::draw_rgb(GridMap &grid_map, Vec3b value, string target_frame, vector<Point2d>& bbox)
{
    tf::Transform tf_base_world = get_base_world();
    for (vector<Point2d>::iterator it = bbox.begin(); it != bbox.end(); it++)
    {
        Point2i bb_m;
        if (target_frame == "world_map")
        {
            Point2d bb_world;
            base2world(*it, bb_world, tf_base_world);
            grid_map.world2map(bb_world, bb_m);
        }
        else if (target_frame == "map")
            grid_map.world2map(*it, bb_m);
        else if (target_frame == "view_map")
        {
            Point2d bb_world;
            base2view(*it, bb_world);
            grid_map.world2map(bb_world, bb_m);
        }
        if (grid_map.is_in_map(bb_m))
        {
            grid_map.map_.at<Vec3b>(bb_m.x, bb_m.y) = value;
        }
    }
}

std::vector<int> Agent::CreateRandomNums(int min,int max, int num)
{
	std::vector<int> res;
	res.clear();
	if (max - min + 1 < num)
	{
		return res;
	}
	// srand((unsigned)time(NULL));
	for (auto i{0}; i < num; i++)
	{
		while (true)
		{
			auto temp{ rand() % (max + 1 - min) + min };
			auto iter{ find(res.begin(),res.end(),temp) };
			if (res.end() == iter)
			{
				res.push_back(temp);
				break;
			}		
		}
	}
    sort(res.begin(), res.end());
	return res;
}

bool Agent::view(GridMap &grid_map)
{   
    if(is_collision_ || is_arrive_){
        return is_collision_;
    }
    is_collision_ = draw(grid_map, -1, "world_map", bbox_);
    // if (is_collision_ > 0)
    //     return true;
    // else
    //     return false;
    Point2d xy0_2d;
    base2view(sensor_base_, xy0_2d);
    Point2i xy0_2i;
    view_map_.world2map(xy0_2d, xy0_2i);
    view_map_.empty_map();
    GridMap laser_map = view_map_;
    tf::Transform tf_view_world = get_view_world();
    for (int i = 0; i < view_map_.img_height_; i++){
        for (int j = 0; j < view_map_.img_width_; j++)
        {
            Point2i index(i, j);
            Point2d xy_view, xy_world, xy_base;
            Point2i xy_map;
            view_map_.map2world(index, xy_view);
            view2base(xy_view, xy_base);
            double view_angle = atan2(xy_base.y - sensor_base_.y, xy_base.x - sensor_base_.x);
            // if (view_angle <= view_angle_begin_ || view_angle >= view_angle_end_ ||
            //         xy_base.x < view_min_dist_ || xy_base.x > view_max_dist_)
            // {
            //     continue;
            // }
            tf::Vector3 xyv_view(xy_view.x, xy_view.y, 0);
            tf::Vector3 xyv_world = tf_view_world * xyv_view;
            xy_world.x = xyv_world.getX();
            xy_world.y = xyv_world.getY();
            grid_map.world2map(xy_world, xy_map);
            if (grid_map.is_in_map(xy_map))
            {
                if (grid_map.map_.at<uchar>(xy_map.x, xy_map.y) < 250)
                {
                    view_map_.map_.at<uchar>(i, j) = 0;
                }
                else if (grid_map.map_.at<uchar>(xy_map.x, xy_map.y) >= 250)
                {
                    view_map_.map_.at<uchar>(i, j) = 255;
                }
            }
        }
    }
    if (use_laser_)
    {
        int angular_map_size = 72;
        double miss_perc = 0.0, hit = 0;
        bool miss_flag = false;
        hits_.clear();
        hit_points_x_.clear();
        hit_points_y_.clear();
        angular_map_.resize(angular_map_size);
        for (int m = 0; m < angular_map_size; m++)
            angular_map_[m] = view_max_dist_;
        double map_width = tf_base_view_.getOrigin().getX();
        double map_height = tf_base_view_.getOrigin().getY();
        double max_range = sqrt(map_width * map_width + map_height * map_height);
        double angle_step = abs(view_angle_end_ - view_angle_begin_) / range_total_;
        double angular_map_step = abs(view_angle_end_ - view_angle_begin_) / angular_map_size;
        std::vector<int> randoms_index = CreateRandomNums(0, range_total_ - 1, miss_perc * range_total_);
        // cout << "************************************" << endl;
        for (int i = 0; i < range_total_; i++)
        {
            double cur_angle = view_angle_begin_ + angle_step * i;
            int angular_map_i = int(angle_step * i / angular_map_step);
            double x = max_range * cos(cur_angle);
            double y = max_range * sin(cur_angle);
            Point2d xy(x, y);
            Point2d xy_view_2d;
            base2view(xy, xy_view_2d);
            Point2i xy_view_2i;
            view_map_.world2map(xy_view_2d, xy_view_2i);
            if(miss_flag){
                if (std::binary_search(randoms_index.begin(), randoms_index.end(), i)){
                    hit = 0.0;
                    // cout << "No:" << i << endl;
                }
                else{
                    double hit = bresenhamLine(xy0_2i.x, xy0_2i.y, xy_view_2i.x, xy_view_2i.y, view_map_, laser_map);
                }
            }
            else{
                double hit = bresenhamLine(xy0_2i.x, xy0_2i.y, xy_view_2i.x, xy_view_2i.y, view_map_, laser_map);
            }
            hits_.push_back(hit);
            if (hit < angular_map_[angular_map_i])
                angular_map_[angular_map_i] = hit;
            hit_points_x_.push_back(hit * cos(cur_angle));
            hit_points_y_.push_back(hit * sin(cur_angle));
        }
        view_map_ = laser_map;
    }
//    if (use_laser_)
//    {
//        double map_width = tf_base_view_.getOrigin().getX();
//        double map_height = tf_base_view_.getOrigin().getY();
//        double max_range = sqrt(map_width * map_width + map_height * map_height);
//        double angle_step = abs(view_angle_end_ - view_angle_begin_) / range_total_;
//        for (int i = 0; i < range_total_; i++)
//        {
//            double cur_angle = view_angle_begin_ + angle_step * i;
//            double x = max_range * cos(cur_angle);
//            double y = max_range * sin(cur_angle);
//            Point2d xy(x, y);
//            Point2d xy_view_2d;
//            base2view(xy, xy_view_2d);
//            double dist = sqrt((xy_view_2d.x - xy0_2d.x) * (xy_view_2d.x - xy0_2d.x) +
//                               (xy_view_2d.y - xy0_2d.y) * (xy_view_2d.y - xy0_2d.y));
//            double dx = (xy_view_2d.x - xy0_2d.x) / dist;
//            double dy = (xy_view_2d.y - xy0_2d.y) / dist;
//            double ds = view_map_.resolution_;
//            int j = 0;
//            bool line_end = false;
//            int end_x = -1;
//            int end_y = -1;
//            while(true)
//            {
//                double line_x = dx * j * ds + xy0_2d.x;
//                double line_y = dy * j * ds + xy0_2d.y;
//                Point2d line_xy(line_x, line_y);
//                Point2i line_map;
//                view_map_.world2map(line_xy, line_map);
//                if (view_map_.is_in_map(line_map))
//                {
//                    int cur_data = view_map_.map_.at<uchar>(line_map.x, line_map.y);
//                    if (line_end == false)
//                    {
//                        if (cur_data == 0)
//                        {
//                            laser_map.map_.at<uchar>(line_map.x, line_map.y) = 0;
//                            line_end = true;
//                            end_x = x;
//                            end_y = y;
//                        }
//                        else
//                        {
//                            laser_map.map_.at<uchar>(line_map.x, line_map.y) = 255;
//                        }
//                    }
//                    else
//                    {
////                        if ((line_map.x != end_x) && (line_map.y != end_y))
//                        {
//                            laser_map.map_.at<uchar>(line_map.x, line_map.y) = 200;

//                        }

//                    }
//                }
//                else
//                    break;
//                j += 1;
//            }
//        }
//        view_map_ = laser_map;
//    }
    draw(view_map_, 100, "view_map", bbox_);
    if (is_collision_ > 0)
        return true;
    else
        return false;
//    return is_collision_;
}

// double Agent::bresenhamLine(int x1, int y1, int x2, int y2, GridMap &source_map, GridMap &target_map)
// {
//     double hit = 6, hit_gau, K, B, theta, delta_x, delta_y, gau_x, gau_y, min_dis;
//     vector<Point2i> vec;
//     Point2d xy0;
//     Point2i min_xy;
//     target_map.map2world(Point2i(x1, y1), xy0);

//     int w = x2 - x1;
//     int h = y2 - y1;
//     int dx = ((w>0)<<1) - 1;
//     int dy = ((h>0)<<1) - 1;
//     w = abs(w);
//     h = abs(h);
//     int f , y , x, delta1,delta2;
//     bool line_end = false;
//     int end_x = -1;
//     int end_y = -1;
//     double mu {0}, sigma {1.0};
//     // cout << "mu: " << mu << "," << "sigma:" << sigma << endl;
//     normal_distribution<> dist {mu, sigma};
//     random_device rd;
//     default_random_engine rng {rd()};
//     if( w > h )
//     {
//         f = 2*h - w;
//         delta1 = 2*h;
//         delta2 = (h-w)*2;
//         for( x = x1 , y = y1 ; x!=x2 ; x += dx )
//         {
//             Point2i cur(x, y);
//             if (source_map.is_in_map(cur))
//             {
//                 vec.push_back(cur);
//                 int cur_data = source_map.map_.at<uchar>(cur.x, cur.y);
//                 if (line_end == false)
//                 {
//                     if (cur_data != 0){
//                         target_map.map_.at<uchar>(cur.x, cur.y) = 255;
//                     }
//                     else if (end_x == -1)
//                     {
//                         // target_map.map_.at<uchar>(cur.x, cur.y) = 0;
//                         // cout << "*****************" << endl;
//                         // cout << "cur: " << cur.x << "," << cur.y << endl;
                        
//                         // cout << dist(rng) << endl;

//                         line_end = true;
//                         // end_x = x;
//                         // end_y = y;

//                         Point2d cur_xy;
//                         target_map.map2world(cur, cur_xy);
//                         // cout << "xy0: " << xy0.x << "," << xy0.y << endl;
//                         // cout << "cur_xy: " << cur_xy.x << "," << cur_xy.y << endl;
//                         hit = sqrt((xy0.x - cur_xy.x) * (xy0.x - cur_xy.x) +
//                                    (xy0.y - cur_xy.y) * (xy0.y - cur_xy.y));
//                         hit_gau = hit + dist(rng);
//                         if (hit_gau < 0)
//                             hit_gau = hit;
//                         // cout << "hit_gau: " << hit_gau <<endl;
//                         K = (xy0.y - cur_xy.y) / (xy0.x - cur_xy.x);
//                         // cout << "K: " << K <<endl;
//                         theta = atan(K);
//                         // cout << "theta: " << theta <<endl;
//                         delta_x = hit_gau * cos(theta);
//                         delta_y = hit_gau * sin(theta);
//                         // cout << "delta_x: " << delta_x <<endl;
//                         // cout << "delta_y: " << delta_y <<endl;
//                         gau_x = xy0.x - delta_x;
//                         gau_y = xy0.y - delta_y;
//                         if (gau_x < 0 || gau_y < 0){
//                             B = gau_y - K * gau_x;
//                             if(K > 1){
//                                 gau_x = - B / K;
//                                 gau_y = 0;
//                             }
//                             else if(K < 1){
//                                 gau_x = 0;
//                                 gau_y = B;
//                             }
//                             else{
//                                 gau_x = 0;
//                                 gau_y = 0;
//                             }
//                         }
//                         // cout << "gau: " << gau_x << "," << gau_y << endl;
//                         Point2d gau(gau_x, gau_y);
//                         Point2i cur_gau;
//                         target_map.world2map(gau, cur_gau);
//                         // cout << "cur_gau: " << cur_gau.x << "," << cur_gau.y << endl;
                            
//                         // target_map.map_.at<uchar>(cur_gau.x, cur_gau.y) = 0;
//                         end_x = cur_gau.x;
//                         end_y = cur_gau.y;

//                         // cout << "hit: " << hit << endl;
//                     }
//                 }
//                 else
//                 {
//                     if ((cur.x != end_x) && (cur.y != end_y)){
//                         target_map.map_.at<uchar>(cur.x, cur.y) = 200;
//                     }

//                 }
//             }
//             else{
//                 break;
//             }
//             if( f < 0 )
//             {
//                 f += delta1;
//             }
//             else
//             {
//                 y += dy;
//                 f += delta2;
//             }
//         }
//         int min_index = 0;
//         min_dis = abs(vec[0].x - end_x) + abs(vec[0].y - end_y);
//         for(int i=1; i < vec.size(); i++){
//             if((abs(vec[i].x - end_x) + abs(vec[i].y - end_y)) < min_dis){
//                 min_dis = abs(vec[i].x - end_x) + abs(vec[i].y - end_y);
//                 min_xy = vec[i];
//                 min_index = i;
//             }
//         }

//         for(int i=0; i < vec.size(); i++){
//             if(i < min_index){
//                 target_map.map_.at<uchar>(vec[i].x, vec[i].y) = 255;
//             }
//             else if(i == min_index){
//                 target_map.map_.at<uchar>(vec[i].x, vec[i].y) = 0;
//             }
//             else if(min_index != 0){
//                 target_map.map_.at<uchar>(vec[i].x, vec[i].y) = 200;
//             }
//         }
//     }
//     else
//     {
//         f = 2*w - h;
//         delta1 = w*2;
//         delta2 = (w-h)*2;
//         for( x = x1 , y = y1 ; y!=y2 ; y += dy )
//         {
//             Point2i cur(x, y);
//             if (source_map.is_in_map(cur))
//             {
//                 vec.push_back(cur);
//                 int cur_data = source_map.map_.at<uchar>(cur.x, cur.y);
//                 if (line_end == false)
//                 {
//                     if (cur_data != 0){
//                         target_map.map_.at<uchar>(cur.x, cur.y) = 255;
//                     }
//                     else if (end_x == -1)
//                     {
//                         // cout << "*****************" << endl;
//                         // cout << "cur: " << cur.x << "," << cur.y << endl;
//                         // target_map.map_.at<uchar>(cur.x, cur.y) = 0;
//                         line_end = true;
//                         // end_x = x;
//                         // end_y = y;

//                         Point2d cur_xy;
//                         target_map.map2world(cur, cur_xy);
//                         // cout << "xy0: " << xy0.x << "," << xy0.y << endl;
//                         // cout << "cur_xy: " << cur_xy.x << "," << cur_xy.y << endl;
//                         hit = sqrt((xy0.x - cur_xy.x) * (xy0.x - cur_xy.x) +
//                                    (xy0.y - cur_xy.y) * (xy0.y - cur_xy.y));
                        
//                         hit_gau = hit + dist(rng);
//                         if (hit_gau < 0)
//                             hit_gau = hit;
//                         // cout << "hit: " << hit << endl;
//                         // cout << "hit_gau: " << hit_gau <<endl;
//                         K = (xy0.y - cur_xy.y) / (xy0.x - cur_xy.x);
//                         // cout << "K: " << K <<endl;
//                         theta = atan(K);
//                         // cout << "theta: " << theta <<endl;
//                         delta_x = hit_gau * cos(theta);
//                         delta_y = hit_gau * sin(theta);
//                         // cout << "delta_x: " << delta_x <<endl;
//                         // cout << "delta_y: " << delta_y <<endl;
//                         gau_x = xy0.x - delta_x;
//                         gau_y = xy0.y - delta_y;
//                         if (gau_x < 0 || gau_y < 0){
//                             B = gau_y - K * gau_x;
//                             if(K > 1){
//                                 gau_x = - B / K;
//                                 gau_y = 0;
//                             }
//                             else if(K < 1){
//                                 gau_x = 0;
//                                 gau_y = B;
//                             }
//                             else{
//                                 gau_x = 0;
//                                 gau_y = 0;
//                             }
//                         }
//                         // cout << "gau: " << gau_x << "," << gau_y << endl;
//                         Point2d gau(gau_x, gau_y);
//                         Point2i cur_gau;
//                         target_map.world2map(gau, cur_gau);
//                         // cout << "cur_gau: " << cur_gau.x << "," << cur_gau.y << endl;
//                         // target_map.map_.at<uchar>(cur_gau.x, cur_gau.y) = 0;
//                         end_x = cur_gau.x;
//                         end_y = cur_gau.y;
//                     }
//                 }
//                 else
//                 {
//                     if ((cur.x != end_x) && (cur.y != end_y)){
//                         target_map.map_.at<uchar>(cur.x, cur.y) = 200;
//                     }

//                 }
//             }
//             else{
//                 break;
//             }
//             if( f < 0 )
//             {
//                 f += delta1;
//             }
//             else
//             {
//                 x += dx;
//                 f += delta2;
//             }
//         }
//         int min_index = 0;
//         min_dis = abs(vec[0].x - end_x) + abs(vec[0].y - end_y);
//         for(int i=1; i < vec.size(); i++){
//             if((abs(vec[i].x - end_x) + abs(vec[i].y - end_y)) < min_dis){
//                 min_dis = abs(vec[i].x - end_x) + abs(vec[i].y - end_y);
//                 min_xy = vec[i];
//                 min_index = i;
//             }
//         }

//         for(int i=0; i < vec.size(); i++){
//             if(i < min_index){
//                 target_map.map_.at<uchar>(vec[i].x, vec[i].y) = 255;
//             }
//             else if(i == min_index){
//                 target_map.map_.at<uchar>(vec[i].x, vec[i].y) = 0;
//             }
//             else if(min_index != 0){
//                 target_map.map_.at<uchar>(vec[i].x, vec[i].y) = 200;
//             }
//         }
//     }
//     return hit;
// }

double Agent::bresenhamLine(int x1, int y1, int x2, int y2, GridMap &source_map, GridMap &target_map)
{
    double hit = 6;
    Point2d xy0;
    target_map.map2world(Point2i(x1, y1), xy0);

    int w = x2 - x1;
    int h = y2 - y1;
    int dx = ((w>0)<<1) - 1;
    int dy = ((h>0)<<1) - 1;
    w = abs(w);
    h = abs(h);
    int f , y , x, delta1,delta2;
    bool line_end = false;
    int end_x = -1;
    int end_y = -1;
    if( w > h )
    {
        f = 2*h - w;
        delta1 = 2*h;
        delta2 = (h-w)*2;
        for( x = x1 , y = y1 ; x!=x2 ; x += dx )
        {
            Point2i cur(x, y);
            if (source_map.is_in_map(cur))
            {
                int cur_data = source_map.map_.at<uchar>(cur.x, cur.y);
                if (line_end == false)
                {
                    if (cur_data != 0)
                        target_map.map_.at<uchar>(cur.x, cur.y) = 255;
                    else if (end_x == -1)
                    {
                        target_map.map_.at<uchar>(cur.x, cur.y) = 0;
                        line_end = true;
                        end_x = x;
                        end_y = y;

                        Point2d cur_xy;
                        target_map.map2world(cur, cur_xy);
                        hit = sqrt((xy0.x - cur_xy.x) * (xy0.x - cur_xy.x) +
                                   (xy0.y - cur_xy.y) * (xy0.y - cur_xy.y));
                    }
                }
                else
                {
                    if ((cur.x != end_x) && (cur.y != end_y))
                        target_map.map_.at<uchar>(cur.x, cur.y) = 200;

                }
            }
            else
                return hit;
            if( f < 0 )
            {
                f += delta1;
            }
            else
            {
                y += dy;
                f += delta2;
            }
        }
    }
    else
    {
        f = 2*w - h;
        delta1 = w*2;
        delta2 = (w-h)*2;
        for( x = x1 , y = y1 ; y!=y2 ; y += dy )
        {
            Point2i cur(x, y);
            if (source_map.is_in_map(cur))
            {
                int cur_data = source_map.map_.at<uchar>(cur.x, cur.y);
                if (line_end == false)
                {
                    if (cur_data != 0)
                        target_map.map_.at<uchar>(cur.x, cur.y) = 255;
                    else if (end_x == -1)
                    {
                        target_map.map_.at<uchar>(cur.x, cur.y) = 0;
                        line_end = true;
                        end_x = x;
                        end_y = y;

                        Point2d cur_xy;
                        target_map.map2world(cur, cur_xy);
                        hit = sqrt((xy0.x - cur_xy.x) * (xy0.x - cur_xy.x) +
                                   (xy0.y - cur_xy.y) * (xy0.y - cur_xy.y));
                    }
                }
                else
                {
                    if ((cur.x != end_x) && (cur.y != end_y))
                        target_map.map_.at<uchar>(cur.x, cur.y) = 200;

                }
            }
            else
                return hit;
            if( f < 0 )
            {
                f += delta1;
            }
            else
            {
                x += dx;
                f += delta2;
            }
        }
    }
    return hit;
}

void Agent::get_corners(double &pax, double &pay, double &pbx, double &pby)
{
   // return agent corners;
   // 都转成世界坐标系
   if (shape_ == "circle"){
       Point2d pa_base(sizes_[0] - sizes_[2], sizes_[1] - sizes_[2]), pb_base(
               sizes_[0] + sizes_[2], sizes_[1] + sizes_[2]), pa_w(pax, pay), pb_w(pbx, pby);

       base2world(pa_base, pa_w, get_base_world());
       base2world(pb_base, pb_w, get_base_world());
       pax = pa_w.x;
       pay = pa_w.y;
       pbx = pb_w.x;
       pby = pb_w.y;
   }
   else if ( shape_ == "rectangle" ){
       Point2d pa_base(sizes_[0], sizes_[2]), pb_base(sizes_[1], sizes_[3]), pa_w(pax, pay), pb_w(pbx, pby);

       base2world(pa_base, pa_w, get_base_world());
       base2world(pb_base, pb_w, get_base_world());
       pax = pa_w.x;
       pay = pa_w.y;
       pbx = pb_w.x;
       pby = pb_w.y;
   }
}

PedAgent::PedAgent(string ktype):Agent(ktype){
    state_ = 0;
    step_len_ = 0.15;
    remaining_dist_ = 0;
    cur_traj_index_ = 0;
}
PedAgent::PedAgent(string ktype, double resolution):Agent(ktype,resolution){
    state_ = 0;
    step_len_ = 0.3;
    remaining_dist_ = 0;
    cur_traj_index_ = 0;
}
// sizes 6维 [0 1 2]: 左腿相对base的x y r [3 4 5] : 右腿相对base的x y r
void PedAgent::init_shape(string shape, const vector<double> &sizes)
{
    if (shape == "leg")
    {
        left_leg_bbox_.clear();
        right_leg_bbox_.clear();
        shape_ = shape;
        sizes_ = sizes;
        left_leg_r_ = sizes_[2];
        right_leg_r_ = sizes_[5];
        const vector<double> left_leg_size = {0, 0, left_leg_r_};
        const vector<double> right_leg_size = {0, 0, right_leg_r_};
        Agent::init_shape_circle(left_leg_size, left_leg_bbox_);
        Agent::init_shape_circle(right_leg_size, right_leg_bbox_);
    }
    else if (shape == "circle" || shape == "rectangle")
    {
        Agent::init_shape(shape, sizes);
    }
}

void PedAgent::set_params(){

}

void PedAgent::set_position(Point3d pose){
    last_robot_pose_ = robot_pose_;
    robot_pose_ = pose;
}

void PedAgent::update_bbox(){
    if (shape_ == "leg"){
        double move_dist = sqrt((robot_pose_.x - last_robot_pose_.x) * (robot_pose_.x - last_robot_pose_.x)
                                + (robot_pose_.y - last_robot_pose_.y) * (robot_pose_.y - last_robot_pose_.y));
        last_state_ = state_;
        state_ = int((move_dist + remaining_dist_) / step_len_ + last_state_);
        remaining_dist_ = move_dist + remaining_dist_ - (state_ - last_state_) * step_len_;
        state_ %= 7;
    //    std::cout << move_dist << ", " << state_ << std::endl;
        if(state_ == 0 || state_ == 4) {
            left_leg_.x = sizes_[0];
            left_leg_.y = sizes_[1];
            left_leg_.z = 0;
            right_leg_.x = sizes_[3];
            right_leg_.y = sizes_[4];
            right_leg_.z = 0;
        }
        // 右脚 领先 左脚半步
        else if(state_ == 1 || state_ == 3){
            left_leg_.x = - step_len_ / 2;
            right_leg_.x =  step_len_ / 2;
        }
        // 右脚 领先 左脚1步
        else if(state_ == 2){
            left_leg_.x = - step_len_;
            right_leg_.x = step_len_;
        }
        // 左脚 领先 右脚半步
        else if(state_ == 5 || state_ == 7){
            left_leg_.x = step_len_ / 2;
            right_leg_.x = - step_len_ / 2;
        }
        // 左脚 领先 右脚1步
        else if(state_ == 6){
            left_leg_.x = step_len_;
            right_leg_.x = - step_len_;
        }
        // 人腿模型 : http://ras.papercept.net/images/temp/IROS/files/0122.pdf
    }
}

bool PedAgent::draw_leg(GridMap &grid_map, int value)
{
    bool is_collision = false;
    tf::Transform tf_base_world = get_base_world();
    tf::Transform tf_leg_base = get_leg_base(left_leg_);
    for (vector<Point2d>::iterator it = left_leg_bbox_.begin(); it != left_leg_bbox_.end(); it++)
    {
        Point2i bb_m;
        Point2d bb_world, bb_base;
        leg2base(*it, bb_base, tf_leg_base);
        base2world(bb_base, bb_world, tf_base_world);
        grid_map.world2map(bb_world, bb_m);
        if (grid_map.is_in_map(bb_m))
        {
            if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 0)
                is_collision = true;
            else if (value >= 0)
                grid_map.map_.at<uchar>(bb_m.x, bb_m.y) = value;
        }
    }
    tf_leg_base = get_leg_base(right_leg_);
    for (vector<Point2d>::iterator it = right_leg_bbox_.begin(); it != right_leg_bbox_.end(); it++)
    {
        Point2i bb_m;
        Point2d bb_world, bb_base;
        leg2base(*it, bb_base, tf_leg_base);
        base2world(bb_base, bb_world, tf_base_world);
        grid_map.world2map(bb_world, bb_m);
        if (grid_map.is_in_map(bb_m))
        {
            if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 1) // 行人占位标记为1
                is_collision = true;
            else if (value >= 0)
                grid_map.map_.at<uchar>(bb_m.x, bb_m.y) = value;
        }
    }
    return is_collision;
}

bool PedAgent::draw_leg_rgb(GridMap &grid_map, Vec3b value)
{
    bool is_collision = false;
    tf::Transform tf_base_world = get_base_world();
    tf::Transform tf_leg_base = get_leg_base(left_leg_);
    for (vector<Point2d>::iterator it = left_leg_bbox_.begin(); it != left_leg_bbox_.end(); it++)
    {
        Point2i bb_m;
        Point2d bb_world, bb_base;
        leg2base(*it, bb_base, tf_leg_base);
        base2world(bb_base, bb_world, tf_base_world);
        grid_map.world2map(bb_world, bb_m);
        if (grid_map.is_in_map(bb_m))
        {
//            if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 0)
//                is_collision = true;
//            else if (value >= 0)
                grid_map.map_.at<Vec3b>(bb_m.x, bb_m.y) = value;
        }
    }
    tf_leg_base = get_leg_base(right_leg_);
    for (vector<Point2d>::iterator it = right_leg_bbox_.begin(); it != right_leg_bbox_.end(); it++)
    {
        Point2i bb_m;
        Point2d bb_world, bb_base;
        leg2base(*it, bb_base, tf_leg_base);
        base2world(bb_base, bb_world, tf_base_world);
        grid_map.world2map(bb_world, bb_m);
        if (grid_map.is_in_map(bb_m))
        {
//            if (grid_map.map_.at<uchar>(bb_m.x, bb_m.y) == 0)
//                is_collision = true;
//            else if (value >= 0)
                grid_map.map_.at<Vec3b>(bb_m.x, bb_m.y) = value;
        }
    }
    return is_collision;
}

tf::Transform PedAgent::get_leg_base(Point3d &leg)
{
    tf::Transform tf_leg_base;
    tf_leg_base.setOrigin(tf::Vector3(leg.x, leg.y, 0));
    tf_leg_base.setRotation(tf::Quaternion(0,0,0,1));
    return tf_leg_base;
}

bool PedAgent::arrive(geometry_msgs::Point p)
{
    if ((p.x - robot_pose_.x) * (p.x - robot_pose_.x) + (p.y - robot_pose_.y) * (p.y - robot_pose_.y) < 0.04)
        return true;
    else
        return false;
}

void PedAgent::leg2base(const Point2d &xy_leg, Point2d &xy_base, tf::Transform tf_leg_base)
{
    tf::Vector3 xyv_leg(xy_leg.x, xy_leg.y, 0);
    tf::Vector3 xyv_base = tf_leg_base * xyv_leg;
    xy_base.x = xyv_base.getX();
    xy_base.y = xyv_base.getY();
}

geometry_msgs::Point PedAgent::_get_cur_goal()
{
    return trajectory_[cur_traj_index_%trajectory_.size()];

}
