#ifndef __gs_common_h_
#define __gs_common_h_
#include <cmath>
#include <cassert>
#include <vector>
#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <stdlib.h>

using namespace std;

#undef max
#undef min

namespace gslam
{
/* template classes and methods for pose calculation */
template<class T>
  struct point
  {
    inline point() :
        x(0), y(0)
    {
    }
    inline point(T _x, T _y) :
        x(_x), y(_y)
    {
    }
    T x, y;
  };

template<class T>
  inline point<T> operator+(const point<T>& p1, const point<T>& p2)
  {
    return point<T>(p1.x + p2.x, p1.y + p2.y);
  }

template<class T>
  inline point<T> operator -(const point<T> & p1, const point<T> & p2)
  {
    return point<T>(p1.x - p2.x, p1.y - p2.y);
  }

template<class T>
  inline point<T> operator *(const point<T>& p, const T& v)
  {
    return point<T>(p.x * v, p.y * v);
  }

template<class T>
  inline point<T> operator *(const T& v, const point<T>& p)
  {
    return point<T>(p.x * v, p.y * v);
  }

template<class T>
  inline T operator *(const point<T>& p1, const point<T>& p2)
  {
    return p1.x * p2.x + p1.y * p2.y;
  }

template<class T, class A>
  struct orientedpoint : public point<T>
  {
    inline orientedpoint() :
        point<T>(0, 0), theta(0)
    {
    }
    ;
    inline orientedpoint(const point<T>& p);
    inline orientedpoint(T x, T y, A _theta) :
        point<T>(x, y), theta(_theta)
    {
    }
    inline void normalize();
    inline orientedpoint<T, A> rotate(A alpha)
    {
      T s = sin(alpha), c = cos(alpha);
      A a = alpha + theta;
      a = atan2(sin(a), cos(a));
      return orientedpoint(c * this->x - s * this->y, s * this->x + c * this->y, a);
    }
    A theta;
  };

template<class T, class A>
  void orientedpoint<T, A>::normalize()
  {
    if (theta >= -M_PI && theta < M_PI)
      return;

    int multiplier = (int)(theta / (2 * M_PI));
    theta = theta - multiplier * 2 * M_PI;
    if (theta >= M_PI)
      theta -= 2 * M_PI;
    if (theta < -M_PI)
      theta += 2 * M_PI;
  }

template<class T, class A>
  orientedpoint<T, A>::orientedpoint(const point<T>& p)
  {
    this->x = p.x;
    this->y = p.y;
    this->theta = 0.;
  }

template<class T, class A>
  orientedpoint<T, A> operator+(const orientedpoint<T, A>& p1, const orientedpoint<T, A>& p2)
  {
    return orientedpoint<T, A>(p1.x + p2.x, p1.y + p2.y, p1.theta + p2.theta);
  }

template<class T, class A>
  orientedpoint<T, A> operator -(const orientedpoint<T, A> & p1, const orientedpoint<T, A> & p2)
  {
    return orientedpoint<T, A>(p1.x - p2.x, p1.y - p2.y, p1.theta - p2.theta);
  }

template<class T, class A>
  orientedpoint<T, A> operator *(const orientedpoint<T, A>& p, const T& v)
  {
    return orientedpoint<T, A>(p.x * v, p.y * v, p.theta * v);
  }

template<class T, class A>
  orientedpoint<T, A> operator *(const T& v, const orientedpoint<T, A>& p)
  {
    return orientedpoint<T, A>(p.x * v, p.y * v, p.theta * v);
  }

template<class T, class A>
  orientedpoint<T, A> absoluteDifference(const orientedpoint<T, A>& p1,const orientedpoint<T, A>& p2)
  {
    orientedpoint<T, A> delta = p1 - p2;
    delta.theta = atan2(sin(delta.theta), cos(delta.theta));
    double s = sin(p2.theta), c = cos(p2.theta);
    return orientedpoint<T, A>(c * delta.x + s * delta.y, -s * delta.x + c * delta.y, delta.theta);
  }

template<class T, class A>
  orientedpoint<T, A> absoluteSum(const orientedpoint<T, A>& p1,const orientedpoint<T, A>& p2)
  {
    double s = sin(p1.theta), c = cos(p1.theta);
    return orientedpoint<T, A>(c * p2.x - s * p2.y, s * p2.x + c * p2.y, p2.theta) + p1;
  }

template<class T, class A>
  point<T> absoluteSum(const orientedpoint<T, A>& p1,const point<T>& p2)
  {
    double s = sin(p1.theta), c = cos(p1.theta);
    return point<T>(c * p2.x - s * p2.y, s * p2.x + c * p2.y) + (point<T> )p1;
  }

template<class T>
  struct pointcomparator
  {
    bool operator ()(const point<T>& a, const point<T>& b) const
    {
      return a.x < b.x || (a.x == b.x && a.y < b.y);
    }
  };

template<class T>
  struct pointradialcomparator
  {
    point<T> origin;
    bool operator ()(const point<T>& a, const point<T>& b) const
    {
      point<T> delta1 = a - origin;
      point<T> delta2 = b - origin;
      return (atan2(delta1.y, delta1.x) < atan2(delta2.y, delta2.x));
    }
  };

template<class T>
  inline point<T> max(const point<T>& p1, const point<T>& p2)
  {
    point<T> p = p1;
    p.x = p.x > p2.x ? p.x : p2.x;
    p.y = p.y > p2.y ? p.y : p2.y;
    return p;
  }

template<class T>
  inline point<T> min(const point<T>& p1, const point<T>& p2)
  {
    point<T> p = p1;
    p.x = p.x < p2.x ? p.x : p2.x;
    p.y = p.y < p2.y ? p.y : p2.y;
    return p;
  }

template<class T, class F>
  inline point<T> interpolate(const point<T>& p1, const F& t1, const point<T>& p2, const F& t2, const F& t3)
  {
    F gain = (t3 - t1) / (t2 - t1);
    point<T> p = p1 + (p2 - p1) * gain;
    return p;
  }

template<class T, class A, class F>
  inline orientedpoint<T, A> interpolate(const orientedpoint<T, A>& p1, const F& t1, const orientedpoint<T, A>& p2,
  const F& t2
                                         , const F& t3)
  {
    F gain = (t3 - t1) / (t2 - t1);
    orientedpoint<T, A> p;
    p.x = p1.x + (p2.x - p1.x) * gain;
    p.y = p1.y + (p2.y - p1.y) * gain;
    double s = sin(p1.theta) + sin(p2.theta) * gain, c = cos(p1.theta) + cos(p2.theta) * gain;
    p.theta = atan2(s, c);
    return p;
  }

template<class T>
  inline double euclidianDist(const point<T>& p1, const point<T>& p2)
  {
    return hypot(p1.x - p2.x, p1.y - p2.y);
  }
template<class T, class A>
  inline double euclidianDist(const orientedpoint<T, A>& p1, const orientedpoint<T, A>& p2)
  {
    return hypot(p1.x - p2.x, p1.y - p2.y);
  }
template<class T, class A>
  inline double euclidianDist(const orientedpoint<T, A>& p1, const point<T>& p2)
  {
    return hypot(p1.x - p2.x, p1.y - p2.y);
  }
template<class T, class A>
  inline double euclidianDist(const point<T>& p1, const orientedpoint<T, A>& p2)
  {
    return hypot(p1.x - p2.x, p1.y - p2.y);
  }

typedef point<int> IntPoint;
typedef point<float> FloatPoint;
typedef point<double> Point;
typedef orientedpoint<double, double> OrientedPoint;

typedef struct
{
  int num_points;
  IntPoint* points;
} GridLineTraversalLine;

struct GridLineTraversal
{
  static void gridLine(IntPoint start, IntPoint end, GridLineTraversalLine *line)
  {
    int dx, dy, incr1, incr2, d, x, y, xend, yend, xdirflag, ydirflag;
    int cnt = 0;

    dx = abs(end.x - start.x);
    dy = abs(end.y - start.y);

    if (dy <= dx)
    {
      d = 2 * dy - dx;
      incr1 = 2 * dy;
      incr2 = 2 * (dy - dx);
      if (start.x > end.x)
      {
        x = end.x;
        y = end.y;
        ydirflag = (-1);
        xend = start.x;
      }
      else
      {
        x = start.x;
        y = start.y;
        ydirflag = 1;
        xend = end.x;
      }
      line->points[cnt].x = x;
      line->points[cnt].y = y;
      cnt++;
      if (((end.y - start.y) * ydirflag) > 0)
      {
        while (x < xend)
        {
          x++;
          if (d < 0)
          {
            d += incr1;
          }
          else
          {
            y++;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      }
      else
      {
        while (x < xend)
        {
          x++;
          if (d < 0)
          {
            d += incr1;
          }
          else
          {
            y--;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      }
    }
    else
    {
      d = 2 * dx - dy;
      incr1 = 2 * dx;
      incr2 = 2 * (dx - dy);
      if (start.y > end.y)
      {
        y = end.y;
        x = end.x;
        yend = start.y;
        xdirflag = (-1);
      }
      else
      {
        y = start.y;
        x = start.x;
        yend = end.y;
        xdirflag = 1;
      }
      line->points[cnt].x = x;
      line->points[cnt].y = y;
      cnt++;
      if (((end.x - start.x) * xdirflag) > 0)
      {
        while (y < yend)
        {
          y++;
          if (d < 0)
          {
            d += incr1;
          }
          else
          {
            x++;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      }
      else
      {
        while (y < yend)
        {
          y++;
          if (d < 0)
          {
            d += incr1;
          }
          else
          {
            x--;
            d += incr2;
          }
          line->points[cnt].x = x;
          line->points[cnt].y = y;
          cnt++;
        }
      }
    }
    line->num_points = cnt;
  }
  inline static void gridLineCore(IntPoint start, IntPoint end, GridLineTraversalLine *line)
  {
    int i, j;
    int half;
    IntPoint v;
    gridLineCore(start, end, line);
    if (start.x != line->points[0].x || start.y != line->points[0].y)
    {
      half = line->num_points / 2;
      for (i = 0, j = line->num_points - 1; i < half; i++, j--)
      {
        v = line->points[i];
        line->points[i] = line->points[j];
        line->points[j] = v;
      }
    }
  }

};
}

#endif  //__gs_common_h_
