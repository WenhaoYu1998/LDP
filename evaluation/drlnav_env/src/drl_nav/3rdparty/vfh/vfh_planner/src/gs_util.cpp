#include "gs_util.h"

int _Line2d::BuildLine(float _x1, float _y1, float _x2, float _y2)
{
  if (_x1 == _x2 && _y1 == _y2)
    return 0;

  x1 = _x1;
  y1 = _y1;
  x2 = _x2;
  y2 = _y2;
  length = _Sqrt2(_x2 - _x1, _y2 - _y1);

  // get two-point form and normal form (rho >= 0)
  if (_Abs(x1 - x2) < 1e-6)
  {
    // s_theta == 0
    a = 0;
    b = (x1 + x2) / 2;
    if (b > 0)
    {
      rho = b;
      theta = 0;
    }
    else
    {
      rho = -b;
      theta = _PI;
    }
  }
  else
  {
    a = (y2 - y1) / (x2 - x1);
    b = y1 - a * x1;
    float x = 0, y = 0;
    rho = _CalcPerpendicular(a, b, x, y);
    if (rho == 0)
      theta = _NormRad(atan2(y2 - y1, x2 - x1) + _PI_2);
    else
      theta = atan2(y, x);
  }
  c_theta = cos(theta);
  s_theta = sin(theta);
  return 1;
}

void _BresenhamLine(int x1, int y1, int x2, int y2, std::vector<gpose> &line, bool clear/* = true*/)
{
  int line_pts[2048][2];
  int dx = x2 - x1;
  int dy = y2 - y1;
  int N = _Max(_Abs(dx), _Abs(dy)) + 1;
  _BresenhamLine(dx, dy, line_pts, N);

  gpose p;
  if(clear)
  {
      line.resize(N);
      gpose *ps = &line[0];
      for (int i = 0; i < N; i++)
      {
        ps[i].x = line_pts[i][0] + x1;
        ps[i].y = line_pts[i][1] + y1;
      }
  }
  else
  {
      for(int i = 0; i < N; i++)
      {
          p.x = line_pts[i][0] + x1;
          p.y = line_pts[i][1] + y1;
          line.push_back(p);
      }
  }
}

void _BresenhamLine(int dx, int dy, int line[][2], int N)
{
  int tmp;
  int absDx = (dx >= 0 ? dx : -dx);
  int absDy = (dy >= 0 ? dy : -dy);
  int incX = (dx >= 0 ? 1 : -1);
  int incY = (dy >= 0 ? 1 : -1);
  bool alongY = (absDx < absDy);
  if (alongY)
  {
    tmp = absDx;
    absDx = absDy;
    absDy = tmp;
    tmp = incX;
    incX = incY;
    incY = tmp;
  }
  int delta, error, resetError;
  delta = 2 * absDy;
  error = -absDx;
  resetError = -2 * absDx;
  for (int x = 0, y = 0, i = 0; i < N; i++)
  {
    line[i][0] = x;
    line[i][1] = y;
    error += delta;
    if (error > 0)
    {
      y += incY;
      error += resetError;
    }
    x += incX;
  }
  if (alongY)
  {
    for (int i = 0; i < N; i++)
    {
      tmp = line[i][0];
      line[i][0] = line[i][1];
      line[i][1] = tmp;
    }
  }
}

int _Line2d::BuildLine(float *x_arr, float *y_arr, int N, float *w_arr /* = 0 */)
{
  if (N < 2)
    return 0;

  // Least Square Method for Linear Fit: Y = aX + b

  std::vector<float> _weight;
  if (!w_arr)
  {
    _weight.assign(N, 1.0f);
    w_arr = &_weight[0];
  }

  float Mx = 0;
  float My = 0;
  float Mxx = 0;
  float Mxy = 0;
  float Myy = 0;
  float Mw = 0;
  for (int i = 0; i < N; i++)
  {
    float x = x_arr[i];
    float y = y_arr[i];
    float w = w_arr[i];
    Mx += x * w;
    My += y * w;
    Mw += w;
    Mxx += x * x * w;
    Mxy += x * y * w;
    Myy += y * y * w;
  }

  float diff = Mw * Mxx - Mx * Mx;
  if (diff != 0)
  {
    // case: y = ax + b

    float a, b;

    // this solution may be ill
    //a = (Mw * Mxy - Mx * My) / diff;
    //b = (My * Mxx - Mx * Mxy) / diff;

    // seek for more finer solution
    Mx /= Mw;
    My /= Mw;
    Mxy /= Mw;
    Mxx /= Mw;
    Myy /= Mw;

    float alpha = Mxy - Mx * My;
    float beta = Mxx - Myy + My * My - Mx * Mx;
    float gamma = -alpha;

    float a1 = (-beta + sqrt(beta * beta - 4 * alpha * gamma)) / (2 * alpha);
    float a2 = (-beta - sqrt(beta * beta - 4 * alpha * gamma)) / (2 * alpha);
    float b1 = My - a1 * Mx;
    float b2 = My - a2 * Mx;

    // select better one from the above two solutions
    // according to total square error with samples
    float e1 = 0, e2 = 0;
    for (int i = 0; i < N; i++)
    {
      e1 += _Square((y_arr[i] - a1 * x_arr[i] - b1) * w_arr[i]);
      e2 += _Square((y_arr[i] - a2 * x_arr[i] - b2) * w_arr[i]);
    }

    e1 /= (a1 * a1 + 1);
    e2 /= (a2 * a2 + 1);
    a = e1 < e2 ? a1 : a2;
    b = e1 < e2 ? b1 : b2;

    // calculate foot of perpendicular of first and last point
    int i1 = 0, i2 = N - 1;
    float _x1 = x_arr[i1];
    float _y1 = y_arr[i1];
    float _x2 = x_arr[i2];
    float _y2 = y_arr[i2];
    _CalcPerpendicular(a, b, _x1, _y1);
    _CalcPerpendicular(a, b, _x2, _y2);

    return BuildLine(_x1, _y1, _x2, _y2);
  }
  else
  {
    // case: x = c
    int i1 = 0, i2 = N - 1;
    return BuildLine(x_arr[i1], y_arr[i1], x_arr[i2], y_arr[i2]);
  }
}

float _Line2d::Similarity(_Line2d &other)
{
  float sim = 0.0f;
  return sim;
}

float _Line2d::Collineation(const _Line2d &other, float d_thtresh /* = 0.3f */, float a_thtresh /* = _ToRad(10.0f) */)
{
  float col = 0.0f;
  float rad_diff = _Abs(_SemiNormRad(theta - other.theta));
  if (rad_diff <= a_thtresh)
  {
    float d1_diff = _Abs(NormalTo(other.x1, other.y1));
    float d2_diff = _Abs(NormalTo(other.x2, other.y2));
    float d3_diff = _Abs(other.NormalTo(x1, y1));
    float d4_diff = _Abs(other.NormalTo(x2, y2));
    float d_diff = d1_diff + d2_diff;
    if (d_diff > d3_diff + d4_diff && d_diff < d_thtresh * 1.5)
      d_diff = d3_diff + d4_diff;
    if (d_diff <= d_thtresh)
      col = 1.0;
  }
  return col;
}

int Corner::BuildCorner(Line2d &_l1, Line2d &_l2, float d_thresh /* = 0.2f */)
{
  l1 = _l1;
  l2 = _l2;
  if (_DiffRad(l1.theta, l2.theta) < 0)
    _Swap(l1, l2);
  _Unclockwise(l1.x1, l1.y1, l1.x2, l1.y2);
  _Unclockwise(l2.x1, l2.y1, l2.x2, l2.y2);
  _CalcIntersection(l1.x1, l1.y1, l1.x2, l1.y2, l2.x1, l2.y1, l2.x2, l2.y2, ox, oy);
  _Plane2Polar(ox, oy, r, theta);
  p1 = l1.Proportion(ox, oy);
  p2 = l2.Proportion(ox, oy);
  type = CT_NONE;
  int pat_l1 = 0, pat_l2 = 0;
  if (_Abs((0 - p1) * l1.length) < d_thresh)
    pat_l1 = 1;
  else if (_Abs((1 - p1) * l1.length) < d_thresh)
    pat_l1 = 3;
  else if (p1 > 0 && p1 < 1)
    pat_l1 = 2;
  else if (p1 > 1)
    pat_l1 = 4;
  if (_Abs((0 - p2) * l2.length) < d_thresh)
    pat_l2 = 1;
  else if (_Abs((1 - p2) * l2.length) < d_thresh)
    pat_l2 = 3;
  else if (p2 > 0 && p2 < 1)
    pat_l2 = 2;
  else if (p2 > 1)
    pat_l2 = 4;
  CornerType pat_tab[5][5] = { {CT_UNKNOWN, CT_UNKNOWN, CT_UNKNOWN, CT_UNKNOWN, CT_HL0}, {CT_UNKNOWN, CT_UNKNOWN, CT_UNKNOWN, CT_L0, CT_UNKNOWN}, {
      CT_HT270, CT_T270, CT_CROSS, CT_UNKNOWN, CT_UNKNOWN},
                              {CT_UNKNOWN, CT_L180, CT_T0, CT_UNKNOWN, CT_UNKNOWN}, {CT_HL180, CT_UNKNOWN, CT_HT0, CT_UNKNOWN, CT_UNKNOWN}, };
  type = pat_tab[pat_l1][pat_l2];
  return 0;
}

void _AStar::build_graph(int width, int height, int *grid, int split, int conn8 /* = 1 */)
{
  clear();
#define GV(x, y) grid[y * W + x]
  W = width;
  H = height;
  S = W * H;
  N = conn8 ? 8 : 4;
  G = new mcell[S];
  P = new int[S];
  // 7 0 4
  // 3 * 1
  // 6 2 5
  int neigh[8][2] = { {0, -1}, {1, 0}, {0, 1}, {-1, 0}, {1, -1}, {1, 1}, {-1, 1}, {-1, -1}};
  for (int y = 0, id = 0; y < H; y++)
  {
    for (int x = 0; x < W; x++)
    {
      mcell *p = &G[id++];
      int v = GV(x, y);
      if (v > split)
      {
        p->x = x;
        p->y = y;
        for (int i = 0; i < 8; i++)
        {
          int xi = x + neigh[i][0];
          int yi = y + neigh[i][1];
          int vi = GV(xi, yi);
          if (i >= N || xi < 0 || xi >= W || yi < 0 || yi >= H || vi <= split)
          {
            p->neigh[i] = -1;
          }
          else
          {
            p->weight[i] = i < 4 ? 1.0f * (256 - vi) : 1.4142f * (256 - vi);
            p->neigh[i] = yi * W + xi;
          }
        }
      }
      else
      {
        p->x = -1;
      }
    }
  }
}

int _AStar::astar_search(int srcx, int srcy, int dstx, int dsty)
{
  L = 0;
  int ox = srcx, oy = srcy, dx = dstx, dy = dsty;
  int oid = oy * W + ox;
  int did = dy * W + dx;
  if (G[oid].x == -1 || G[did].x == -1)
  {
    return -1;
  }
  for (int i = 0; i < S; i++)
    G[i].s = -1;
  int ln = 0, head, getit = 0;
  G[oid].p = oid;
  G[oid].d = 0;
  G[oid].s = 0;
  P[ln++] = oid;
  do
  {
    head = P[0];
    if (head == did)
      break;
    ln--;
    G[head].s = 1;
    for (int i = 0; i < ln; i++)
      P[i] = P[i + 1];
    for (int i = 0; i < N; i++)
    {
      int nei = G[head].neigh[i];
      mcell *pc = G + nei;
      if (nei != -1)
      {
        float dn = G[head].weight[i];
        if (pc->s == -1)
        {
          pc->s = 0;
          pc->p = head;
          pc->d = G[head].d + dn;
          pc->c = pc->d + sqrt((pc->x - dx) * (pc->x - dx) * 1.0 + (pc->y - dy) * (pc->y - dy) * 1.0);
          int j = ln - 1;
          while (j >= 0)
          {
            if (G[P[j]].c <= pc->c)
              break;
            else
            {
              P[j + 1] = P[j];
            }
            j--;
          }
          P[j + 1] = nei;
          ln++;
        }
        else if (pc->s == 0)
        {
          if (pc->d > G[head].d + dn)
          {
            pc->c += G[head].d + dn - pc->d;
            pc->d = G[head].d + dn;
            pc->p = head;
          }
        }
      }
    }
  } while (ln > 0);

  if (head == did)
  {
    int pl = 0;
    for (int i = did; G[i].p != i; i = G[i].p)
    {
      P[pl++] = i;
    }
    P[pl++] = oid;
    for (int i = 0; i < pl / 2; i++)
    {
      int tmp = P[i];
      P[i] = P[pl - 1 - i];
      P[pl - 1 - i] = tmp;
    }
    L = pl;
    return pl;
  }
  else
  {
    return 0;
  }
}
