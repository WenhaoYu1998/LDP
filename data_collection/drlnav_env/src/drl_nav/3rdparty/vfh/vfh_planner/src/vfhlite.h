#include <vector>
#include <math.h>
#include "gs_util.h"
#include "gs_common.h"
using namespace gslam;
class VFH
{
public:
    VFH() {}
    float param_a, param_b, param_c;
    float max_v, min_v, max_rv;
    float robot_radius, safe_dist;
    int param_mu1, param_mu2, param_mu3;
    float sector_angle, window_size;
    float wide_angle;
    float threshold_high, threshold_low;
    int clutter_const;
    float m_delta;

    float prev_ang;
    float vscale;

    int sectors;

    std::vector<float> orig_hist;
    std::vector<bool> binary_hist;
    std::vector<bool> masked_hist;

    std::vector<float> vfh_obj_theta;

    struct vfhcell
    {
      int valid;
      float dist, theta, gamma;
      float mag;
      std::vector<int> locs;
    };

    struct MapIndix{
        int x;
        int y;
    };

    struct vfhwindow
    {
      int glen, gnum, center;
      std::vector<vfhcell> cells_;
      vfhcell *cells;
      void initWindow(float win_size, float grid_size, float rob_radius, float safe_dist, int n_sectors);
    };


    void setVFHParam(float _param_a, float _param_b, float _param_c, float _max_v, float _min_v,
                     float _max_rv, float _robot_radius, float _safe_dist,
                     int _param_mu1, int _param_mu2, int _param_mu3,
                     float _sector_angle, float _window_size, float _wide_angle,
                     float _threshold_high, float _threshold_low, int _clutter_const);
    void set_delta(float delta)
    {
        m_delta = delta;
    }
    inline void setWindowSize(float winsz)
    {
      window_size = winsz;
    }
    float angleWeight(float cand, float goal, float curr, float prev);
    Point steer(const OrientedPoint &robot_pose, const Point &goal, float *lasd = 0);
};
