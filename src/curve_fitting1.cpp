
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

template <typename T> struct LinearFunctor : public std::unary_function<T, T> {
  LinearFunctor(std::vector<T> &c) : m_c(c) {}
  T operator()(T x) { return m_c[0] * x + m_c[1]; }

private:
  vector<T> m_c;
};

struct LinearCostFunctor {
  LinearCostFunctor(double x, double y) : m_x(x), m_y(y) {}
  template <typename T> bool operator()(const T *const x, T *residual) const {
    std::vector<T> c{x[0], x[1]};
    LinearFunctor<T> lFunc(c);
    residual[0] = static_cast<T>(m_y) - lFunc(static_cast<T>(m_x));
    return true;
  }

private:
  const double m_x, m_y;
};

template <typename T> struct NoiseAddingFunctor : public unary_function<T, T> {
  NoiseAddingFunctor(T mean, T stdv) : gen(), dist(mean, stdv) {
    random_device rd;
    gen.seed(rd());
  }
  NoiseAddingFunctor(const NoiseAddingFunctor &obj) {
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

  // 1. Setting Up Sample Points.
  double minx = -5.0, maxx = 5.0, resox = 0.1;
  int num_x = (maxx - minx) / resox + 1;
  // Determine Coefficients
  vector<double> coeff{1.5, 2};

  // 2. Problem Formulation
  vector<double> x_elems(num_x, 0.0), y_elems(num_x, 0.0),
      y_n_elems(num_x, 0.0), y_est_elems(num_x, 0.0);
  {
    // Create X Vector
    iota(x_elems.begin(), x_elems.end(), 0.0);
    transform(x_elems.begin(), x_elems.end(), x_elems.begin(),
              [num_x, minx, resox](double val) { return val * resox + minx; });

    // Create Y Vector
    LinearFunctor<double> lFunc(coeff);
    transform(x_elems.begin(), x_elems.end(), y_elems.begin(),
              [&lFunc](double val) { return lFunc(val); });

    // Create Noised Y Vector
    NoiseAddingFunctor<double> nNoise(0.0, 3.0);
    transform(y_elems.begin(), y_elems.end(), y_n_elems.begin(),
              [&nNoise](double val) { return nNoise(val); });
  }

  // 3. Ceres Estimation
  vector<double> co(2, 0.0);
  {
    ceres::Problem problem;
    for (size_t i = 0; i < x_elems.size(); i++) {
      double x = x_elems[i];
      double y = y_n_elems[i];
      ceres::CostFunction *cost_function =
          new ceres::AutoDiffCostFunction<LinearCostFunctor, 1, 2>(
              new LinearCostFunctor(x, y));
      problem.AddResidualBlock(cost_function, nullptr, co.data());
    }

    // Run the solver
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Status Report
    std::cout << summary.BriefReport() << std::endl;

    std::cout << "Original Parameters" << std::endl;
    {
      std::ostream_iterator<double> out(std::cout, ", ");
      copy(coeff.begin(), coeff.end(), out);
      std::cout << std::endl << std::flush;
    }

    std::cout << "Estimated Parameters" << std::endl;
    {
      std::ostream_iterator<double> out(std::cout, ", ");
      copy(co.begin(), co.end(), out);
      std::cout << std::endl << std::flush;
    }

    // Create Estimated Y Vector
    LinearFunctor<double> lFunc(co);
    transform(x_elems.begin(), x_elems.end(), y_est_elems.begin(),
              [&lFunc](double val) { return lFunc(val); });
  }

  // 4. Plot Result
  {
    plt::title("Original vs Estimated");
    plt::plot(x_elems, y_elems);
    plt::scatter(x_elems, y_n_elems);
    plt::plot(x_elems, y_est_elems);
    plt::show();
  }

  return 0;
}