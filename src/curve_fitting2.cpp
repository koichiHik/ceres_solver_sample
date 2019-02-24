
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

template <typename T> struct CubicFunctor : public unary_function<T, T> {
  CubicFunctor(vector<double> &c) : m_c(c) {}
  T operator()(T x) {
    T x3 = x * x * x;
    T x2 = x * x;
    T x1 = x;
    return m_c[0] * x3 + m_c[1] * x2 + m_c[2] * x1 + m_c[3];
  }

private:
  vector<T> m_c;
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
    T val = x[0] * std::pow(m_x, 3.0) + x[1] * std::pow(m_x, 2.0) + x[2] * m_x +
            x[3];
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

  // 1. Setting Up Sample Points.
  double minx = -5.0, maxx = 5.0, reso = 0.1;
  int num = (maxx - minx) / reso + 1;
  // Determine Coefficient
  vector<double> coeff{0.5, -0.5, -10.5, 1.0};

  // Create X Vector
  vector<double> x_vec(num, 0.0);
  iota(x_vec.begin(), x_vec.end(), 0.0);
  transform(x_vec.begin(), x_vec.end(), x_vec.begin(),
            [minx, reso](double val) { return val * reso + minx; });

  // Create Y Vector
  vector<double> y_vec(num, 0.0);
  transform(x_vec.begin(), x_vec.end(), y_vec.begin(),
            CubicFunctor<double>(coeff));

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

    std::cout << "Original Parameters" << std::endl;
    {
      ostream_iterator<double> out(std::cout, ", ");
      copy(coeff.begin(), coeff.end(), out);
      std::cout << std::endl << std::flush;
    }

    std::cout << "Estimated Parameters" << std::endl;
    {
      ostream_iterator<double> out(std::cout, ", ");
      copy(coeff.begin(), coeff.end(), out);
      std::cout << std::endl << std::flush;
    }

    // Generate Point on Estimated Curve.
    transform(x_vec.begin(), x_vec.end(), y_est.begin(),
              CubicFunctor<double>(co_est));
  }

  // 3. Plot Result
  {
    plt::xlim(minx - 5, maxx + 5);

    plt::plot(x_vec, y_vec);
    plt::title("Orignal vs Estimated");
    plt::scatter(x_vec, y_vec_noise);
    plt::plot(x_vec, y_est);

    plt::show();
  }

  return 0;
}