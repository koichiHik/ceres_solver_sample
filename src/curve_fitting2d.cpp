
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

template <typename T> struct DescritizeFunctor : public unary_function<T, T> {
  DescritizeFunctor(T min, T max, T reso)
      : m_min(min), m_max(max), m_reso(reso) {}
  T operator()(const T &x) const { return x * m_reso + m_min; }

private:
  double m_min, m_max, m_reso;
};

template <typename T> struct CubicFunctor : public unary_function<T, T> {
  CubicFunctor(T c3, T c2, T c1, T c0)
      : m_c3(c3), m_c2(c2), m_c1(c1), m_c0(c0) {}
  T operator()(T x) {
    T x3 = x * x * x;
    T x2 = x * x;
    T x1 = x;
    return m_c3 * x3 + m_c2 * x2 + m_c1 * x1 + m_c0;
  }

private:
  T m_c3, m_c2, m_c1, m_c0;
};

template <typename T> struct SquareFunctor : public unary_function<T, T> {
  SquareFunctor(T c2, T c1, T c0) : m_c2(c2), m_c1(c1), m_c0(c0) {}
  T operator()(T x) { return m_c2 * x * x + m_c1 * x + m_c0; }

private:
  T m_c2, m_c1, m_c0;
};

template <typename T> struct NoiseAddingFunctor : public unary_function<T, T> {
  NoiseAddingFunctor(T mean, T stdv) : engine(), dist(mean, stdv) {}
  T operator()(T x) { return x + dist(engine); }

private:
  default_random_engine engine;
  normal_distribution<> dist;
};

struct CostFunctor {
  CostFunctor(double x, double y) : m_x(x), m_y(y) {}

  template <typename T> bool operator()(const T *const x, T *residual) const {
    T val = x[3] * std::pow(m_x, 3.0) + x[2] * std::pow(m_x, 2.0) + x[1] * m_x +
            x[0];
    residual[0] = val - m_y;
    return true;
  }

private:
  const double m_x;
  const double m_y;
};

int main(int argc, char **argv) {
  std::cout << __FILE__ << std::endl;
  google::InitGoogleLogging(argv[0]);

  // 1. Generate Sample Points.
  double minx = -5.0, maxx = 5.0, reso = 0.1;
  int num = (maxx - minx) / reso + 1;

  // Create X Vector
  vector<double> x_vec(num, 0.0);
  iota(x_vec.begin(), x_vec.end(), 0.0);
  transform(x_vec.begin(), x_vec.end(), x_vec.begin(),
            DescritizeFunctor<double>(minx, maxx, reso));

  // Create Y Vector
  vector<double> y_vec(num, 0.0);
  transform(x_vec.begin(), x_vec.end(), y_vec.begin(),
            CubicFunctor<double>(0.5, -0.5, -10.5, 1.0));

  // Create Noised Y Vector
  vector<double> y_vec_noise(num, 0.0);
  transform(y_vec.begin(), y_vec.end(), y_vec_noise.begin(),
            NoiseAddingFunctor<double>(0.0, 5.0));

  // 2. Create Ceres Problem
  vector<double> co_est(4, 0.0);
  std::vector<double> y_est(x_vec);
  {
    ceres::Problem problem;
    for (size_t i = 0; i < x_vec.size(); i++) {
      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<CostFunctor, 1, 4>(
              new CostFunctor(x_vec[i], y_vec_noise[i]));
      problem.AddResidualBlock(cost_function, nullptr, co_est.data());
    }

    // Run the solver!
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Status Report!
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "a : 0.0, b : 0.0, c : 0.0, d : 0.0" << std::endl;
    std::cout << "a : " << co_est[3] << " b : " << co_est[2]
              << " c: " << co_est[1] << " d : " << co_est[0] << std::endl;

    // Generate Point on Estimated Curve.
    transform(x_vec.begin(), x_vec.end(), y_est.begin(),
              CubicFunctor<double>(co_est[3], co_est[2], co_est[1], co_est[0]));
  }

  // 3. Plot Result
  plt::title("Fitting Sample");
  plt::xlim(minx - 5, maxx + 5);
  // plt::ylim(-50, 150);
  plt::plot(x_vec, y_vec);
  plt::plot(x_vec, y_est);
  plt::scatter(x_vec, y_vec_noise);
  plt::show();

  return 0;
}