
// Standard Lib
#include <iostream>
#include <random>

// STL
#include <algorithm>
#include <iterator>
#include <vector>

// Ceres
#include <ceres/ceres.h>
#include <glog/logging.h>

// Matplot Lib
#include <matplotlibcpp.h>

using namespace std;
namespace plt = matplotlibcpp;

template <typename T> struct CubicFunctor 
  : public std::binary_function<T, T, T> {
  CubicFunctor(std::vector<std::vector<T>> &c)
      : m_c(c) {}
  T operator()(T x, T y) {

    // 3rd order terms
    T x3 = m_c[0][3] * x * x * x;  
    T y1x2 = m_c[1][2] * y * x * x;
    T y2x1 = m_c[2][1] * y * y * x;
    T y3 = m_c[3][0] * y * y * y;  

    // 2nd order terms
    T x2 = m_c[0][2] * x * x;
    T y1x1 = m_c[1][1] * y * x;
    T y2 = m_c[2][0] * y * y;

    // 1st order terms
    T x1 = m_c[0][1] * x;
    T y1 = m_c[1][0] * y;

    // 0th order tems
    T x0y0 = m_c[0][0] * 1.0;

    return x3 + y1x2 + y2x1 + y3 + x2 + y1x1 + y2 + x1 + y1 + x0y0;
  }
private:
  std::vector<std::vector<T>> m_c;
};

template <typename T> struct DescritizeFunctor : public unary_function<T, T> {
  DescritizeFunctor(T min, T max, T reso)
      : m_min(min), m_max(max), m_reso(reso) {}
  T operator()(const T &x) const { return x * m_reso + m_min; }

private:
  double m_min, m_max, m_reso;
};

template <typename T> struct NoiseAddingFunctor : public unary_function<T, T> {
  NoiseAddingFunctor(T mean, T stdv) : gen(), dist(mean, stdv) {
    random_device rd;
    gen.seed(rd());
  }
  NoiseAddingFunctor(const NoiseAddingFunctor& obj) {
    dist = obj.dist;
    gen = obj.gen;
    random_device rd;
    gen.seed(rd());
  }
  T operator()(T x) { return x + dist(gen); }

private:
  mt19937_64 gen;
  normal_distribution<> dist;
};

int main(int argc, char **argv) {
  std::cout << __FILE__ << std::endl;

  // 0. Determine Coefficient
  std::vector<std::vector<double>> coeff
  { {1.0, -1.0, 0.5, -0.2},
    {-1.0, 1.0, 1.0, 0.0}, 
    {0.5, 1.0, 0.0, 0.0},
    {-0.2, 0.0, 0.0, 0.0} };

  // 1. Setting Up Sample Points.
  double minx = -5.0, maxx = 5.0, resox = 0.5;
  double miny = -5.0, maxy = 5.0, resoy = 0.5;
  int num_x = (maxx - minx) / resox + 1;
  int num_y = (maxy - miny) / resoy + 1;

  // Create X Vector
  vector<vector<double>> x_elems(num_y);
  transform(x_elems.begin(), x_elems.end(), x_elems.begin(),
      [num_x, minx, maxx, resox](const vector<double> &v) {
        vector<double> vnew(num_x, 0.0);
        iota(vnew.begin(), vnew.end(), 0.0);
        transform(vnew.begin(), vnew.end(), vnew.begin(),
          DescritizeFunctor<double>(minx, maxx, resox));
        return vnew;
      });

  // Create Y Vector
  int cnt = 0;
  vector<vector<double>> y_elems(num_y);
  transform(y_elems.begin(), y_elems.end(), y_elems.begin(),
      [num_x, miny, maxy, resoy, &cnt](const vector<double> &v) {
        double val = resoy * cnt + miny;
        cnt++;
        vector<double> vnew(num_x, val);
        return vnew;
      });

  // Create Z Vector
  vector<vector<double>> z_elems(num_y);
  CubicFunctor<double> cFunc(coeff);
  for (int j = 0; j < num_y; j++) {
    for (int i = 0; i < num_x; i++) {
      double x = x_elems[j][i];
      double y = y_elems[j][i];
      z_elems[j].push_back(cFunc(x, y));
    }
  }

  NoiseAddingFunctor<double> nFunc(0.0, 10.0);
  for_each(z_elems.begin(), z_elems.end(), 
    [&](vector<double>& v) {
      transform(v.begin(), v.end(), v.begin(), nFunc);
      //for_each(v.begin(), v.end(), nFunc);
    });

  // Ceres Estimation
  {

  }

  // X. Plot Result
  plt::plot_surface(x_elems, y_elems, z_elems);
  plt::show();

  return 0;
}
