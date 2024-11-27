#include "vfhlite.h"
#include "ros/ros.h"

void VFH::vfhwindow::initWindow(float win_size, float grid_size, float rob_radius, float safe_dist, int n_sectors)
{
  glen = (int)ceil(win_size / grid_size);
  gnum = glen * glen;
  center = glen / 2;
  cells_.resize(gnum);
  cells = &cells_[0];
  vfhcell *cel = cells;
  for (int y = 0; y < glen; y++)
  {
    for (int x = 0; x < glen; x++)
    {
      cel->dist = sqrt(float((y - center) * (y - center) + (x - center) * (x - center))) * grid_size;
      cel->mag = 0;
      if (cel->dist >= rob_radius && cel->dist <= win_size / 2)
      {
        cel->theta = atan2(float(y - center), float(x - center));
        float xx = (safe_dist + rob_radius) / cel->dist;
        if (xx > 0.9)
          xx = 0.9;
        cel->gamma = asin(xx);
        cel->valid = 1;
        cel->locs.clear();
        float low = cel->theta - cel->gamma;
        float high = cel->theta + cel->gamma;
        int idx1 = round(low / 6.2832 * n_sectors) + n_sectors / 2;
        int idx2 = round(high / 6.2832 * n_sectors) + n_sectors / 2;
        for (int i = idx1; i <= idx2; i++)
          cel->locs.push_back((i + n_sectors) % n_sectors);
      }
      else
      {
        cel->theta = cel->gamma = 0;
        cel->valid = 0;
      }
      cel++;
    }
  }
}

void VFH::setVFHParam(float _param_a, float _param_b, float _param_c, float _max_v, float _min_v, float _max_rv, float _robot_radius,
                          float _safe_dist, int _param_mu1, int _param_mu2, int _param_mu3, float _sector_angle, float _window_size,
                          float _wide_angle, float _threshold_high, float _threshold_low,
                          int _clutter_const)
{
  param_a = _param_a;
  param_b = _param_b;
  param_c = _param_c;
  max_v = _max_v;
  min_v = _min_v;
  max_rv = _max_rv;
  robot_radius = _robot_radius;
  safe_dist = _safe_dist;
  param_mu1 = _param_mu1;
  param_mu2 = _param_mu2;
  param_mu3 = _param_mu3;
  sector_angle = _sector_angle;
  window_size = _window_size;
  wide_angle = _wide_angle;
  threshold_high = _threshold_high;
  threshold_low = _threshold_low;
  clutter_const = _clutter_const;
  sectors = static_cast<int>(6.2832 / _sector_angle), orig_hist.resize(sectors);
  binary_hist.resize(sectors);
  masked_hist.resize(sectors);
  for (int s = 0; s < sectors; ++s)
  {
    orig_hist[s] = 0.0f;
    //binary_hist[s] = masked_hist[s] = false;
  }

  prev_ang = 0.0f;
  vscale = 1.0f;
}

float VFH::angleWeight(float cand, float goal, float curr, float prev)
{
  return _Abs(_NormRad(cand - goal)) * param_mu1 + _Abs(_NormRad(cand - curr)) * param_mu2 + _Abs(_NormRad(cand - prev)) * param_mu3;
}

Point VFH::steer(const OrientedPoint &robot_pose, const Point &goal, float *lasd/* = 0*/)
{
  // build histogram
  for (int s = 0; s < sectors; ++s)
    orig_hist[s] = 0.0f;

  vfhwindow vfhw;
  vfhw.initWindow(window_size, m_delta, robot_radius, safe_dist, sectors);

  if (lasd)
  {
    const int RES = 1440 / 2;
    float delta = 6.28318f / RES;
    float la = 0;
    for (int i = 0; i < RES; i++)
    {
      if (lasd[i] < window_size / 2)
      {
        int ix = int(lasd[i] * cos(la) / m_delta) + vfhw.center;
        int iy = int(lasd[i] * sin(la) / m_delta) + vfhw.center;
        vfhcell *cel = &vfhw.cells[iy * vfhw.glen + ix];
        double a_dif = _Abs(_NormRad(la - robot_pose.theta));
        if (cel->mag == 0)
        {
          float val = 255;
          double dist_factor = a_dif < param_b ? (param_c - ((param_c - 1) / param_b) * a_dif) : 1;
          dist_factor *= param_a / pow(cel->dist, 3);
          cel->mag = dist_factor;
          int num = cel->locs.size();
          if (num)
          {
            int *locs = &cel->locs[0];
            for (int i = 0; i < num; i++)
            {
              orig_hist[locs[i]] += cel->mag;
            }
          }
        }
      }

      la += delta;
    }
  }

  for (int s = 0; s < sectors; ++s)
  {
  //  if (orig_hist[s] > threshold_high)
    if (orig_hist[s] > threshold_low)
      binary_hist[s] = true;
    else if (orig_hist[s] < threshold_low)
      binary_hist[s] = false;
  }

  std::vector<std::pair<int, int> > openings;
  std::pair<int, int> currr;
  bool in_opening = false;

  vfh_obj_theta.clear();

  for (int i = 0; i < sectors; ++i)
  {
    if (!in_opening && !binary_hist[i])
    {
      in_opening = true;
      currr.first = i;
    }
    else if (in_opening && binary_hist[i])
    {
      currr.second = i;
      in_opening = false;
      openings.push_back(currr);
    }
    if (in_opening && !binary_hist[i])
    {
      float angle = i * sector_angle  - _PI;
      vfh_obj_theta.push_back(angle);
    }
  }

  if (in_opening)
  {
    currr.second = sectors - 1;
    openings.push_back(currr);
  }

  int lastidx = openings.size() - 1;
  if (lastidx >= 1 && openings[0].first == 0 && openings[lastidx].second == sectors - 1)
  {
    openings[0].first = openings[lastidx].first - sectors;
    openings[lastidx].second = openings[0].second + sectors;
  }

  float goal_ang = atan2f(goal.y - robot_pose.y, goal.x - robot_pose.x);
  float curr = robot_pose.theta;

  int idx = 0;
  float best = goal_ang;
  float cost = 9999;
  float half = wide_angle / 2;

  while (idx < openings.size())
  {
    float opening_width = (openings[idx].second - openings[idx].first + 1) * sector_angle;
    float small_ang;
    float large_ang;
    if (_Abs(opening_width - _PI2) < 0.05)
    { // no obstacle
      small_ang = -_PI;
      large_ang = _PI;
    }
    else
    {
      small_ang = openings[idx].first * sector_angle + half - _PI;
      large_ang = openings[idx].second * sector_angle - half - _PI;
    }

    if (opening_width > wide_angle)
    {
      float r_angle = _NormRad(openings[idx].first * sector_angle + half - _PI);
      float l_angle = _NormRad(openings[idx].second * sector_angle - half - _PI);
      float l_cost = angleWeight(l_angle, goal_ang, curr, prev_ang);
      float r_cost = angleWeight(r_angle, goal_ang, curr, prev_ang);
      float g_cost = angleWeight(goal_ang, goal_ang, curr, prev_ang);
      if (goal_ang > small_ang && goal_ang < large_ang && g_cost < cost)
      {
        cost = g_cost;
        best = goal_ang;
      }
      if (l_cost < cost)
      {
        cost = l_cost;
        best = l_angle;
      }
      if (r_cost < cost)
      {
        cost = r_cost;
        best = r_angle;
      }
    }
    else
    {
      float center, c_cost;
      if (openings[idx].first < 0)
        center = _NormRad((openings[idx].first - openings[idx].second) * sector_angle / 2.0 - _PI);
      else
        center = _NormRad((openings[idx].first + openings[idx].second) * sector_angle / 2.0 - _PI);

      c_cost = angleWeight(center, goal_ang, curr, prev_ang);

      if (c_cost < cost)
      {
        best = center;
        cost = c_cost;
      }
    }
    idx++;
  }
  float move_ang = best;
  int sector = (int((move_ang + _PI) / sector_angle) + sectors) % sectors;

  float clutter = orig_hist[sector];
  float ang = _NormRad(move_ang + robot_pose.theta);
  clutter = std::min(clutter, static_cast<float>(clutter_const));
  float scale = 2 * _Abs(ang);
  float v = max_v / scale * sqrtf(1.0 - clutter / clutter_const);
  float vel = std::min(v, max_v);
  prev_ang = _NormRad(move_ang + robot_pose.theta);

  vel *= vscale;
//  ROS_INFO("%lf %lf", move_ang, ang);
  return Point(vel, prev_ang);
}
