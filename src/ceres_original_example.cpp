
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

// Ceres
#include "glog/logging.h"
#include <ceres/ceres.h>

namespace plt = matplotlibcpp;

struct Point2D {
  Point2D() : m_x(0.0), m_y(0.0) {}
  Point2D(double x, double y) : m_x(x), m_y(y) {}
  double m_x;
  double m_y;
};

struct CirclePntGenerator {
  CirclePntGenerator(int num, double r) : m_total(num), m_num(0), m_r(r) {}
  void operator()(Point2D &pnt) {
    pnt.m_x = m_r * std::cos(2.0 * M_PI * static_cast<double>(m_num) / m_total);
    pnt.m_y = m_r * std::sin(2.0 * M_PI * static_cast<double>(m_num) / m_total);
    m_num++;
  }
  int m_total, m_num;
  double m_r;
};

struct RandomNoiseAddFunctor {
  RandomNoiseAddFunctor(double avg, double stdv)
      : engine(0), norm_dist(avg, stdv) {}
  void operator()(Point2D &pnt) {
    double r = std::sqrt(std::pow(pnt.m_x, 2.0) + std::pow(pnt.m_y, 2.0));
    pnt.m_x += r * norm_dist(engine);
    pnt.m_y += r * norm_dist(engine);
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
                [&x](const T &pnt) { x.push_back(pnt.m_x); });
  std::for_each(vec2d.begin(), vec2d.end(),
                [&y](const T &pnt) { y.push_back(pnt.m_y); });

  // plt::plot(x, y);
  plt::scatter(x, y, 1);

  return true;
}

struct OdomDiffResidual {
  OdomDiffResidual(double x_diff, double y_diff) : 
    m_x_diff(x_diff), m_y_diff(y_diff) {}

  template <typename T>
  bool operator()(const T *const diff_x, const T *const diff_y, T *residual) const {
    residual[0] = (m_x_diff - diff_x[0]) * (m_x_diff - diff_x[0])
          + (m_y_diff - diff_y[0]) * (m_y_diff - diff_y[0]);
    return true;
  }
  double m_x_diff, m_y_diff;
};

int main(int argc, char **argv) {
  std::cout << __FILE__ << std::endl;
  std::cout << "Ceres Optimization Example" << std::endl;
  google::InitGoogleLogging(argv[0]);

  const int pnt_num = 360;
  std::vector<Point2D> circle_pnts(pnt_num);
  std::for_each(circle_pnts.begin(), circle_pnts.end(),
                CirclePntGenerator(pnt_num, 3.0));
  

  std::vector<Point2D> diff;
  {
    Point2D last_pnt(0.0, 0.0);
    diff.push_back(last_pnt);
    for (auto pnt : circle_pnts) {
      if (last_pnt.m_x == 0.0 && last_pnt.m_y == 0.0) {
        last_pnt = pnt;
        continue;
      }
      diff.push_back(Point2D(pnt.m_x - last_pnt.m_x, pnt.m_y - last_pnt.m_y));
      last_pnt = pnt;
    }
  }

  std::vector<Point2D> noised_diff(diff);
  std::for_each(noised_diff.begin(), noised_diff.end(),
                RandomNoiseAddFunctor(0, 2.0));

  std::vector<Point2D> odom;
  {
    Point2D last_pnt(3.0, 0.0);
    for (auto odo_diff : noised_diff) {
      //std::cout << "x_diff : " << odo_diff.m_x << "y_diff : " << odo_diff.m_y << std::endl;
      last_pnt.m_x += odo_diff.m_x;
      last_pnt.m_y += odo_diff.m_y;
      odom.push_back(last_pnt);
    }
  }

  // Add all residual block
  ceres::Problem problem;
  std::vector<Point2D> refined_odom(noised_diff);
  Point2D start(odom.front().m_x, odom.front().m_y);
  Point2D end(odom.back().m_x, odom.back().m_y);
  double loop_x = end.m_x - start.m_x;
  double loop_y = end.m_y - start.m_y;
  {
    for (auto odo_diff : refined_odom) {
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<OdomDiffResidual, 1, 1, 1>(
            new OdomDiffResidual(odo_diff.m_x, odo_diff.m_y)),
            nullptr, &odo_diff.m_x, &odo_diff.m_y);
    }
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<OdomDiffResidual, 1, 1, 1>(
          new OdomDiffResidual(0, 0)),
          nullptr, &loop_x, &loop_y);
  }
  
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  // Solve!
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Result
  std::cout << summary.BriefReport() << std::endl;


  plt::xlim(-5, 5);
  plt::ylim(-5, 5);
  plot(circle_pnts);
  plot(refined_odom);
  //plot(odom);
  plt::show();

  return 0;
}