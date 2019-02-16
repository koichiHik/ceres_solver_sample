
// Standard Library
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

// STL
#include <algorithm>
#include <iterator>
#include <vector>

// MATPLOTLIB
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

struct Point2D {
  double x;
  double y;
};

struct CirclePntGenerator {
  CirclePntGenerator(int num, double r) : m_total(num), m_num(0), m_r(r) {}
  void operator()(Point2D &pnt) {
    pnt.x = m_r * std::cos(2.0 * M_PI * static_cast<double>(m_num) / m_total);
    pnt.y = m_r * std::sin(2.0 * M_PI * static_cast<double>(m_num) / m_total);
    m_num++;
  }
  int m_total, m_num;
  double m_r;
};

struct RandomNoiseAddFunctor {
  RandomNoiseAddFunctor(double avg, double stdv)
      : engine(0), norm_dist(avg, stdv) {}
  void operator()(Point2D &pnt) {
    pnt.x += norm_dist(engine);
    pnt.y += norm_dist(engine);
  }
  std::default_random_engine engine;
  std::normal_distribution<double> norm_dist;
};

template <typename T> bool plot(std::vector<T> &vec2d) {
  std::vector<double> x;
  std::vector<double> y;
  x.reserve(vec2d.size());
  y.reserve(vec2d.size());
  std::for_each(vec2d.begin(), vec2d.end(),
                [&x](const T &pnt) { x.push_back(pnt.x); });
  std::for_each(vec2d.begin(), vec2d.end(),
                [&y](const T &pnt) { y.push_back(pnt.y); });

  // plt::plot(x, y);
  plt::scatter(x, y);

  return true;
}

int main(int, char **) {
  std::cout << __FILE__ << std::endl;
  std::cout << "Ceres Optimization Example" << std::endl;

  const int pnt_num = 360;
  std::vector<Point2D> circle_pnts(pnt_num);
  std::for_each(circle_pnts.begin(), circle_pnts.end(),
                CirclePntGenerator(pnt_num, 3.0));
  std::vector<Point2D> noised_cirle_pnts(circle_pnts);
  std::for_each(noised_cirle_pnts.begin(), noised_cirle_pnts.end(),
                RandomNoiseAddFunctor(0, 0.3));
  plot(circle_pnts);
  plot(noised_cirle_pnts);
  plt::show();

  return 0;
}