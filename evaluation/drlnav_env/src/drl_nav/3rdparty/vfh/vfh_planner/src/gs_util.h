#ifndef __SlamUtil_h_
#define __SlamUtil_h_

#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h>

const float _PI = 3.1415926f;
const float _PI2 = _PI * 2;
const float _PI_2 = _PI / 2;
const float _PI_4 = _PI / 4;

template<class T>
  inline T _Abs(T x)
  {
    return (x >= 0 ? x : -x);
  }
template<class T>
  inline T _Min(T a, T b)
  {
    return (a <= b ? a : b);
  }
template<class T>
  inline T _Max(T a, T b)
  {
    return (a >= b ? a : b);
  }
template<class T>
  inline void _Swap(T &a, T &b)
  {
    T t = a;
    a = b;
    b = t;
  }
template<class T>
  inline void _Chop(T &a, T v)
  {
    if (a > v)
      a = v;
  }
template<class T>
  inline void _Hoe(T &a, T v)
  {
    if (a < v)
      a = v;
  }
template<class T>
  inline void _Crop(T &a, T v1, T v2)
  {
    if (a < v1)
      a = v1;
    else if (a > v2)
      a = v2;
  }
template<class T>
  inline bool _WithIn(T a, T v1, T v2)
  {
    return (a >= v1 && a <= v2);
  }
template<class T>
  inline T _Between(T a, T v1, T v2)
  {
      return (a < v1 ? v1 : (a > v2 ? v2 : a));
  }
template<class T>
  inline T _Square(T x)
  {
    return (x * x);
  }
template<class T>
  inline T _Square2(T x, T y)
  {
    return (x * x + y * y);
  }
template<class T>
  inline T _Square3(T x, T y, T z)
  {
    return (x * x + y * y + z * z);
  }
template<class T>
  inline T _Sqrt2(T x, T y)
  {
    return sqrt(x * x + y * y);
  }
template<class T>
  inline T _Sqrt3(T x, T y, T z)
  {
    return sqrt(x * x + y * y + z * z);
  }
template<class T>
  inline T _Distance2(T x1, T y1, T x2, T y2)
  {
    return _Sqrt2(x2 - x1, y2 - y1);
  }
template<class T>
  inline T _ToRad(T d)
  {
    return (T)(d * _PI / 180);
  }
template<class T>
  inline T _ToDeg(T r)
  {
    return (T)(r / _PI * 180);
  }
template<class T>
  inline bool _IsOrthogonal(T r1, T r2)
  {
    return (_Abs(cos(r1 - r2)) < T(0.000001));
  }
template<class T>
  inline T _Min(T *arr, int N)
  {
    int min_val = arr[0];
    for (int i = 1; i < N; i++)
      if (arr[i] < min_val)
        min_val = arr[i];
    return min_val;
  }
template<class T>
  inline T _Max(T *arr, int N)
  {
    T max_val = arr[0];
    for (int i = 1; i < N; i++)
      if (arr[i] > max_val)
        max_val = arr[i];
    return max_val;
  }

template<class T>
  inline void _MaxMin(T *arr, int N, T &_max, T &_min)
  {
    _max = _min = arr[0];
    for (int i = 1; i < N; i++)
    {
      if (arr[i] > _max)
        _max = arr[i];
      else if (arr[i] < _min)
        _min = arr[i];
    }
  }

template<class T>
  inline int _Find(T *arr, int N, const T &v)
  {
    for (int i = 0; i < N; i++)
      if (arr[i] == v)
        return i;
    return -1;
  }

template<class T>
  inline void _Plane2Polar(T x, T y, T &r, T &theta)
  {
    r = _Sqrt2(x, y);
    theta = atan2(x, y);
  }

template<class T>
  inline void _Polar2Plane(T r, T theta, T &x, T &y)
  {
    x = r * cos(theta);
    y = r * sin(theta);
  }

// normalize to (-_PI, _PI]
template<class T>
  inline T _NormRad(T rad)
  {
    if (rad <= _PI && rad > -_PI)
      return rad;
    T nrad = rad - ((int)(rad / _PI2)) * _PI2;
    if (nrad > _PI)
      nrad -= _PI2;
    else if (nrad <= -_PI)
      nrad += _PI2;
    return nrad;
  }
// normalize to (-_PI_2, _PI_2]
template<class T>
  inline T _SemiNormRad(T rad)
  {
    if (rad <= _PI_2 && rad > -_PI_2)
      return rad;
    T nrad = rad - ((int)(rad / _PI)) * _PI;
    if (nrad > _PI_2)
      nrad -= _PI;
    else if (nrad <= -_PI_2)
      nrad += _PI;
    return nrad;
  }

template<class T>
  inline T _DiffRad(T from, T to)
  {
    T diff = _NormRad(to - from);
    if (diff > _PI)
      diff -= _PI2;
    else if (diff <= -_PI)
      diff += _PI2;
    return diff;
  }

template<class T>
  inline T _SemiDiffRad(T from, T to)
  {
    T diff = _SemiNormRad(to - from);
    if (diff > _PI_2)
      diff -= _PI;
    else if (diff <= -_PI_2)
      diff += _PI;
    return diff;
  }

template<class T>
  inline bool _Unclockwise(T &x1, T &y1, T &x2, T &y2)
  {
    if ((x1 == 0 && y1 == 0) || (x2 == 0 && y2 == 0))
      return false;
    if (_DiffRad(atan2(y1, x1), atan2(y2, x2)) < 0)
    {
      _Swap(x1, x2);
      _Swap(y1, y2);
      return true;
    }
    return false;
  }

template<class T>
  T _AvgRad(T *rad_arr, int N, T *weight_arr = 0)
  {
    std::vector<T> v_weight;
    if (!weight_arr)
    {
      v_weight.resize(N, 1.0);
      weight_arr = &v_weight[0];
    }
    T rad_n0 = _NormRad(rad_arr[0]);
    bool rot = (rad_n0 > _PI_2 || rad_n0 < -_PI_2);
    rad_arr[0] = weight_arr[0] = 0;
    for (int i = 1; i < N; i++)
    {
      T rad_ni = _NormRad(rad_arr[i]);
      if (rot && rad_ni < 0)
        rad_ni += _PI2;
      rad_arr[0] += rad_ni * weight_arr[i];
      weight_arr[0] += weight_arr[i];
    }
    rad_arr[0] /= weight_arr[0];
    rad_arr[0] = _NormRad(rad_arr[0]);
    return rad_arr[0];
  }

template<class T>
  T _SemiAvgRad(T *rad_arr, int N, T *weight_arr = 0)
  {
    std::vector<T> v_weight;
    if (!weight_arr)
    {
      v_weight.resize(N, 1.0);
      weight_arr = &v_weight[0];
    }
    T rad_n0 = _SemiNormRad(rad_arr[0]);
    bool rot = (rad_n0 > _PI_4 || rad_n0 < -_PI_4);
    rad_arr[0] = weight_arr[0] = 0;
    for (int i = 1; i < N; i++)
    {
      T rad_ni = _SemiNormRad(rad_arr[i]);
      if (rot && rad_ni < 0)
        rad_ni += _PI;
      rad_arr[0] += rad_ni * weight_arr[i];
      weight_arr[0] += weight_arr[i];
    }
    rad_arr[0] /= weight_arr[0];
    rad_arr[0] = _SemiNormRad(rad_arr[0]);
    return rad_arr[0];
  }

template<class T>
  inline T _CalcPerpendicular(T a, T b, T &x, T &y)
  {
    T _x = x, _y = y;
    x = (_x + a * (_y - b)) / (a * a + 1);
    y = a * x + b;
    return _Sqrt2(_x - x, _y - y);
  }

template<class T>
  inline bool _CalcIntersection(T x1, T y1, T x2, T y2, T x3, T y3, T x4, T y4, T &ox, T &oy)
  {
    T dx1 = x2 - x1;
    T dy1 = y2 - y1;
    T dx2 = x4 - x3;
    T dy2 = y4 - y3;
    T diff = dx1 * dy2 - dx2 * dy1;
    if (diff == 0)
      return false;
    T l1 = dx1 * dx1 + dy1 * dy1;
    T l2 = dx2 * dx2 + dy2 * dy2;
    T factor1 = T(((x3 - x1) * dy2 - (y3 - y1) * dx2) * 1.0f / diff);
    T factor2 = T(((x3 - x1) * dy1 - (y3 - y1) * dx1) * 1.0f / diff);
    ox = T(x1 + (dx1 * factor1));
    oy = T(y1 + (dy1 * factor1));
    return true;
  }

template<class _Tcmp, class _Tdat>
  void _SortEx(_Tcmp *cmp_arr, int N, _Tdat *dat_arr = 0, bool decremental = true)
  {
    // select sort
    if (decremental)
    {
      for (int i = 0; i < N; i++)
      {
        int sel = i;
        for (int j = i + 1; j < N; j++)
          if (cmp_arr[j] > cmp_arr[sel])
            sel = j;
        if (sel != i)
        {
          _Swap(cmp_arr[sel], cmp_arr[i]);
          if (dat_arr)
            _Swap(dat_arr[sel], dat_arr[i]);
        }
      }
    }
    else
    {
      for (int i = 0; i < N; i++)
      {
        int sel = i;
        for (int j = i + 1; j < N; j++)
          if (cmp_arr[j] < cmp_arr[sel])
            sel = j;
        if (sel != i)
        {
          _Swap(cmp_arr[sel], cmp_arr[i]);
          if (dat_arr)
            _Swap(dat_arr[sel], dat_arr[i]);
        }
      }
    }
  }

template<class T>
  int _UbietyLine(T a11, T a12, T a21, T a22)
  {
    // 0 to
    // 1 on (outer)
    // 2 intersect
    // 3 on (inner)
    // 4 coincidence
    // 5 in
    if (a11 > a12)
      _Swap(a11, a12);
    if (a21 > a22)
      _Swap(a21, a22);
    int e1 = (a11 == a21 ? 1 : (a11 == a22 ? 3 : (a11 < a21 ? 0 : (a11 > a22 ? 4 : 2))));
    int e2 = (a12 == a21 ? 1 : (a12 == a22 ? 3 : (a12 < a21 ? 0 : (a12 > a22 ? 4 : 2))));
    const int _tab[5][5] = { {0, 1, 2, 3, 5}, {9, 1, 3, 4, 3}, {9, 9, 5, 3, 2}, {9, 9, 9, 1, 1}, {9, 9, 9, 9, 0}};
    return _tab[e1][e2];
  }

template<class T>
  int _OverlapRect(T x11, T y11, T x12, T y12, T x21, T y21, T x22, T y22)
  {
    // -1 to
    // 0 on
    // 1 in
    int u1 = _UbietyLine(x11, x12, x21, x22);
    int u2 = _UbietyLine(y11, y12, y21, y22);
    const int _tab[6][6] = { {0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 1, 1}, {0, 1, 2, 2, 2, 2}, {0, 1, 2, 2, 2, 2}, {0, 1, 2, 2, 2, 2}, {0, 1, 2, 2, 2, 2}};
    return (_tab[u1][u2] - 1);
  }

template<class T>
  inline void _InitArr(T *arr, int N, T val)
  {
    for (int i = 0; i < N; i++)
      arr[i] = val;
  }

template<class T>
  inline void _FreeObj(T * &obj)
  {
    if (obj)
      delete obj;
    obj = 0;
  }
template<class T>
  inline void _FreeArr(T * &arr)
  {
    if (arr)
      delete[] arr;
    arr = 0;
  }
template<class T>
  inline void _FreeObjArr(T **&arr, int len)
  {
    for (int i = 0; i < len; i++)
      delete arr[i];
    delete[] arr;
    arr = 0;
  }
template<class T>
  inline void _FreeArrObj(T **arr, int len)
  {
    for (int i = 0; i < len; i++)
    {
      delete arr[i];
      arr[i] = 0;
    }
  }
template<class T>
  void _FreeVecPtr(std::vector<T *> &vec)
  {
    for (int i = 0; i < (int)vec.size(); i++)
      if (vec[i])
        delete vec[i];
  }

template<class T>
  class _Dual
  {
  public:
    inline _Dual() :
        x(0), y(0)
    {
    }
    inline _Dual(T x, T y) :
        x(x), y(y)
    {
    }
    inline void Set(T _x, T _y)
    {
      x = _x;
      y = _y;
    }
    inline _Dual ToPolar()
    {
      return _Dual(_Sqrt2(x, y), atan2(y, x));
    }
    inline _Dual ToPlane()
    {
      return _Dual(x * cos(y), x * sin(y));
    }
    T x, y;
  };

template<class T1, class T2>
  class _Trin
  {
  public:
    inline _Trin() :
        x(0), y(0), z(0)
    {
    }
    inline _Trin(T1 x, T1 y, T2 z) :
        x(x), y(y), z(z)
    {
    }
    inline void Set(T1 _x, T1 _y, T2 _z)
    {
      x = _x;
      y = _y;
      z = _z;
    }
    // @return = *this + b(diff)
    inline _Trin operator +(const _Trin &b) const
    {
      const T2 c = cos(z), s = sin(z);
      return _Trin(T1(x + b.x * c - b.y * s), T1(y + b.x * s + b.y * c), _NormRad(z + b.z));
    }
    // @return(diff) = *this - b  (b + @return = *this)
    inline _Trin operator -(const _Trin &b) const
    {
      const T2 c = cos(b.z), s = sin(b.z);
      return _Trin(T1((x - b.x) * c + (y - b.y) * s), T1((b.x - x) * s + (y - b.y) * c), _NormRad(z - b.z));
    }
    // @return = *this - b(diff) (@return + b = *this)
    inline _Trin MinusDiff(const _Trin &b) const
    {
      const T2 rot = _NormRad(z - b.z);
      const T2 c = cos(rot), s = sin(rot);
      return _Trin(x - c * b.x + s * b.y, y - c * b.y - s * b.x, rot);
    }
    inline _Trin& operator +=(const _Trin &b)
    {
      const T2 c = cos(z), s = sin(z);
      x += T1(b.x * c - b.y * s);
      y += T1(b.x * s + b.y * c);
      z = _NormRad(z + b.z);
      return *this;
    }
    inline _Trin& operator -=(const _Trin &b)
    {
      const T2 c = cos(b.z), s = sin(b.z);
      T1 tx = T1((x - b.x) * c + (y - b.y) * s);
      T1 ty = T1((b.x - x) * s + (y - b.y) * c);
      x = tx;
      y = ty;
      z = _NormRad(z - b.z);
      return *this;
    }
    inline _Dual<T1> operator +(const _Dual<T1> &b) const
    {
      const T2 c = cos(z), s = sin(z);
      return _Dual<T1>(T1(x + b.x * c - b.y * s), T1(y + b.x * s + b.y * c));
    }
    inline _Dual<T1> CalcMove(const _Dual<T1> &b) const
    {
      T1 _x = b.x - x;
      T1 _y = b.y - y;
      const T2 c = cos(-z), s = sin(-z);
      return _Dual<T1>(T1(_x * c - _y * s), T1(_x * s + _y * c));
    }

    T1 x, y;
    T2 z;
  };

inline int _RandInt(int a, int b)
{
  return (rand() % (b - a) + a);
}

inline float _RandReal(float a, float b)
{
  return ((rand() * 1.0f / RAND_MAX) * (b - a) + a);
}

template<class T>
  class _Size
  {
  public:
    T width, height;
  };

typedef _Dual<int> gpose;
typedef _Dual<int> Point2i;
typedef _Dual<float> Point2d;
typedef _Trin<float, float> Point3d;
typedef _Trin<float, float> Pose2d;
typedef _Trin<float, float> Coor2d;

typedef std::vector<Point2d> P2dArr;
typedef std::vector<Point3d> P3dArr;

template<class T>
  class _Rect
  {
  public:
    T width, height;
  };

template<class T>
  int _PointToRect(const _Dual<T> &p, const _Dual<T> &_r1, const _Dual<T> &_r2, float &dist)
  {
    // 7 6 5
    // 8 0 4
    // 1 2 3
    Point2d r1 = _r1;
    Point2d r2 = _r2;
    if (r1.x > r2.x)
      _Swap(r1.x, r2.x);
    if (r1.y > r2.y)
      _Swap(r1.y, r2.y);
    if (p.x < r1.x)
    {
      if (p.y < r1.y)
      {
        dist = _Distance2(p.x, p.y, r1.x, r1.y);
        return 1;
      }
      else if (p.y > r2.y)
      {
        dist = _Distance2(p.x, p.y, r1.x, r2.y);
        return 7;
      }
      else
      {
        dist = r1.x - p.x;
        return 8;
      }
    }
    else if (p.x > r2.x)
    {
      if (p.y < r1.y)
      {
        dist = _Distance2(p.x, p.y, r2.x, r1.y);
        return 3;
      }
      else if (p.y > r2.y)
      {
        dist = _Distance2(p.x, p.y, r2.x, r2.y);
        return 5;
      }
      else
      {
        dist = p.x - r2.x;
        return 4;
      }
    }
    else
    {
      if (p.y < r1.y)
      {
        dist = r1.y - p.y;
        return 2;
      }
      else if (p.y > r2.y)
      {
        dist = p.y - r2.y;
        return 6;
      }
      else
      {
        float d2 = p.y - r1.y;
        float d4 = r2.x - p.x;
        float d6 = r2.y - p.y;
        float d8 = p.x - r1.x;
        dist = -_Min(_Min(d2, d6), _Min(d4, d8));
        if (dist == -d2)
          return 2;
        if (dist == -d4)
          return 4;
        if (dist == -d6)
          return 6;
        if (dist == -d8)
          return 8;
      }
    }
    return 0;
  }

template<class T>
  inline int _LineToRect(const _Dual<T> &l1, const _Dual<T> &l2, const _Dual<T> &r1, const _Dual<T> &r2, _Dual<T> &dist)
  {
    return 0;
  }

template<class T>
  inline void _CoordTrans2(T ox, T oy, T r, T &x, T &y)
  {
    T c = cos(r);
    T s = sin(r);
    T tx = x * c - y * s;
    T ty = x * s + y * c;
    x = tx + ox;
    y = ty + oy;
  }

template<class T>
  inline void _CoordTrans2(T ox, T oy, T r, _Dual<T> &point)
  {
    T c = cos(r);
    T s = sin(r);
    T tx = point.x * c - point.y * s;
    T ty = point.x * s + point.y * c;
    point.x = tx + ox;
    point.y = ty + oy;
  }

template<class T>
  inline void _CoordTrans2(const _Trin<T, T> &coord, T &x, T &y)
  {
    T c = cos(coord.z);
    T s = sin(coord.z);
    T tx = x * c - y * s;
    T ty = x * s + y * c;
    x = tx + coord.x;
    y = ty + coord.y;
  }

template<class T>
  inline void _CoordTrans2(const _Trin<T, T> &coord, _Dual<T> &point)
  {
    T c = cos(coord.z);
    T s = sin(coord.z);
    T tx = point.x * c - point.y * s;
    T ty = point.x * s + point.y * c;
    point.x = tx + coord.x;
    point.y = ty + coord.y;
  }

void _BresenhamLine(int dx, int dy, int line[][2], int N);
void _BresenhamLine(int x1, int y1, int x2, int y2, std::vector<gpose> &line, bool clear = true);

class _Line2d;
typedef _Line2d Line2d;

class _Line2d
{
  // 2D Linear Equation
  // Two-point Form:	T(x, y) = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1) = 0
  // Normal Form:		N(x, y) = y * sin(theta) + x * cos(theta) - rho = 0
  // Slope-intercept Form:	y = ax + b | x = b
  //
public:
  inline _Line2d()
  {
  }
  inline _Line2d(float rho, float theta)
  {
    BuildLine(rho, theta);
  }
  inline _Line2d(float a, float b, bool y_parallel)
  {
    BuildLine(a, b, y_parallel);
  }
  inline _Line2d(float _x1, float _y1, float _x2, float _y2)
  {
    BuildLine(_x1, _y1, _x2, _y2);
  }

  float x1, y1;
  float x2, y2;
  float length;

  float rho, theta;
  float c_theta;
  float s_theta;

  float a, b;

  int BuildLine(float rho, float theta)
  {
    float c = cos(theta);
    float s = sin(theta);
    float norm_x = rho * cos(theta);
    float norm_y = rho * sin(theta);
    return BuildLine(norm_x, norm_y, norm_x - s, norm_y + c);
  }
  int BuildLine(float a, float b, bool y_parallel)
  {
    if (y_parallel)
      return BuildLine(b, 0, b, 1);
    else
      return BuildLine(0, b, (a == 0 ? 1 : -b / a), (a == 0 ? b : 0));
  }
  int BuildLine(float _x1, float _y1, float _x2, float _y2);
  int BuildLine(float *x_arr, float *y_arr, int num, float *w_arr = 0);
  float Collineation(const _Line2d &other, float d_thtresh = 0.3f, float a_thtresh = _ToRad(10.0f));
  float Similarity(_Line2d &other);
  void Translate(const Pose2d &coord)
  {
    float _x1 = x1, _y1 = y1;
    float _x2 = x2, _y2 = y2;
    _CoordTrans2(coord, _x1, _y1);
    _CoordTrans2(coord, _x2, _y2);
    BuildLine(_x1, _y1, _x2, _y2);
  }
  inline float Proportion(float x, float y) const
  {
    return (x1 == x2) ? (y - y1) / (y2 - y1) : (x - x1) / (x2 - x1);
  }
  inline bool Radiation(float bearing, float &x, float &y) const
  {
    if (_IsOrthogonal(bearing, theta))
      return false;
    x = y = 0;
    Perpendicular(x, y);
    bearing -= theta;
    float d = rho * tan(bearing);
    x -= d * sin(theta);
    y += d * cos(theta);
    return true;
  }
  inline float NormalTo(float x, float y) const
  {
    // return N(x, y)
    return (y * s_theta + x * c_theta - rho);
  }
  inline float Perpendicular(float &x, float &y) const
  {
    float _x = x, _y = y;
    if (x1 == x2)
      x = b;
    else
    {
      x = (_x + a * (_y - b)) / (a * a + 1);
      y = a * x + b;
    }
    return NormalTo(_x, _y);
  }
};

class Corner
{
public:
  enum CornerType
  {
    CT_NONE, CT_UNKNOWN, CT_L0, CT_L180, CT_T0, CT_T270, CT_HL0, CT_HL180, CT_HT0, CT_HT270, CT_CROSS
  } type;

  inline Corner() :
      type(CT_NONE)
  {
  }
  inline Corner(Line2d &_l1, Line2d &_l2, float d_thresh = 0.2f)
  {
    BuildCorner(_l1, _l2, d_thresh);
  }
  int BuildCorner(Line2d &_l1, Line2d &_l2, float d_thresh = 0.2f);
  Line2d l1, l2;
  float ox, oy;
  float r, theta;
  float p1, p2;
};

class _AStar
{
public:

  _AStar()
  {
    G = NULL;
    P = NULL;
    L = 0;
  }

  ~_AStar()
  {
    clear();
  }
  void clear()
  {
    if (G)
    {
      delete[] G;
      G = 0;
    }
    if (P)
    {
      delete[] P;
      P = 0;
    }
  }
  struct mcell
  {
    int x, y, neigh[8];
    int p, s;
    float d, c, weight[8];
  };

  // value >= split, reachable
  void build_graph(int width, int height, int *grid, int split, int conn8 = 1);
  int astar_search(int srcx, int srcy, int dstx, int dsty);

  int W, H, S, N, L;
  int *P;
  mcell *G;
};

namespace _pid
{
class PID
{
private:
  double mP;
  double mI;
  double mD;
  double mError_Sum;
  double mError_Last;
  double mError_Last2;

public:
  PID()
  {
    mP = mI = mD = 0.0;
    Reset();
  }
  inline void SetParameter(double p, double i, double d)
  {
    mP = p;
    mI = i;
    mD = d;
  }
  inline void Reset()
  {
    mError_Sum = mError_Last = mError_Last2 = 0.0;
  }

  inline double Control(double current_value, double desired_value)
  {
    double error = desired_value - current_value;
    mError_Sum += error;
    double dError = mError_Last - mError_Last2;
    mError_Last2 = mError_Last;
    mError_Last = error;
    return mP * error + mI * mError_Sum + mD * dError;
  }
};

const int max_pid_len = 256;
class PIDX
{
private:
  double mP, mI, mD;
  double mV[max_pid_len];
  int mCycleI, mCycleD, mLen, mPtr;

public:
  PIDX()
  {
    mP = mI = mD = 0.0;
    mCycleI = mCycleD = 0;
    mLen = 0;
    Reset();
  }

  inline void Reset()
  {
    mPtr = 0;
  }

  inline void SetParameter(double p, double i, double d, int cycle_i, int cycle_d)
  {
    mP = p;
    mI = i;
    mD = d;
    mCycleI = cycle_i;
    mCycleD = cycle_d;
    if (mCycleI > 254)
      mCycleI = 254;
    if (mCycleD > 254)
      mCycleD = 254;
    mLen = mCycleI > mCycleD ? mCycleI : mCycleD;
    mPtr = 0;
  }

  inline double Control(double current_value, double desired_value)
  {
    double error = desired_value - current_value;
    if (mPtr >= mLen)
    {
      for (int i = 1; i < mPtr; i++)
        mV[i - 1] = mV[i];
      mPtr--;
    }
    if (mPtr < 0)
      return 0.0;
    mV[mPtr++] = error;

    int head = mPtr - mCycleI;
    if (head < 0)
      head = 0;
    double vi = 0.0;
    for (int i = head; i < mPtr; i++)
      vi += mV[i];

    head = mPtr - mCycleD;
    if (head < 0)
      head = 0;
    double vd = mV[mPtr - 1] - mV[head];

    return mP * error + mI * vi + mD * vd;
  }
};

class PIDTracker
{
private:
  PID mC;
  double mP, mV;
  double mMaxSpeed;
  double mMaxAcc;

public:
  PIDTracker()
  {
    mMaxSpeed = 1e6;
    mMaxAcc = 1e6;
  }

  inline void InitPID(double p, double i, double d)
  {
    mC.SetParameter(p, i, d);
  }
  inline void Reset()
  {
    mC.Reset();
  }
  inline void InitPosition(double pos)
  {
    mP = pos;
    mV = 0.0;
  }
  inline void SetMaxSpeed(double max_speed)
  {
    mMaxSpeed = max_speed;
  }
  inline void SetMaxAcc(double max_acc)
  {
    mMaxAcc = max_acc;
  }
  inline double Control(double desired)
  {
    const double t = 0.1;
    const double t2 = t * t * 0.5;
    double a = mC.Control(mP, desired);
    _Crop(a, -mMaxAcc, mMaxAcc);
    mP += mV * t + a * t2;
    mV += a * t;
    _Crop(mV, -mMaxSpeed, mMaxSpeed);
    return mP;
  }
};

class PIDMTRobot
{
private:
  PID mC;
  double mV;
  double mA;
public:
  inline void InitPID(double p, double i, double d)
  {
    mC.SetParameter(p, i, d);
  }
  inline void Reset()
  {
    mC.Reset();
  }
  inline void InitPosition(double para)
  {
    mV = para;
    mA = 0.0;
  }
  inline void Control_Smooth(double &c, double &ret, double da_range)
  {
    const double t = 0.1;
    double da = mC.Control(mV, c);
    _Crop(da, -da_range, da_range);
    mV += mA * t + da * t * t * 0.5;
    mA += da;
    _Crop(mA, -0.1, 0.1);
    ret = mV;
  }
  inline void Control_Tracker(double &c, double &desired)
  {
    const double t = 0.1;
    double da = mC.Control(mV, c);
    mV += mA * t + da * t * t * 0.5;
    mA += da * t;
    desired = mV;
  }
};
}
;

#endif  //__SlamUtil_h_
