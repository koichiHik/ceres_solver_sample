
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

template <typename T>
struct CubicFunctor : public std::binary_function<T, T, T>
{
  CubicFunctor(std::vector<T> &c) : m_c(c) {}
  T operator()(T x, T y)
  {

    // 3rd order terms
    T x3 = m_c[0] * x * x * x;
    T y1x2 = m_c[1] * y * x * x;
    T y2x1 = m_c[2] * y * y * x;
    T y3 = m_c[3] * y * y * y;

    // 2nd order terms
    T x2 = m_c[4] * x * x;
    T y1x1 = m_c[5] * y * x;
    T y2 = m_c[6] * y * y;

    // 1st order terms
    T x1 = m_c[7] * x;
    T y1 = m_c[8] * y;

    // 0th order tems
    T x0y0 = m_c[9] * 1.0;

    return x3 + y1x2 + y2x1 + y3 + x2 + y1x1 + y2 + x1 + y1 + x0y0;
  }

private:
  std::vector<T> m_c;
};

template <typename T>
struct NoiseAddingFunctor : public unary_function<T, T>
{
  NoiseAddingFunctor(T mean, T stdv) : gen(), dist(mean, stdv)
  {
    random_device rd;
    gen.seed(rd());
  }
  NoiseAddingFunctor(const NoiseAddingFunctor &obj)
  {
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

struct CubicCostFunctor
{
  CubicCostFunctor(double x, double y, double z) : m_x(x), m_y(y), m_z(z) {}

  template <typename T>
  bool operator()(const T *const x, T *residual) const
  {
    std::vector<T> c{x[0], x[1], x[2], x[3], x[4],
                     x[5], x[6], x[7], x[8], x[9]};
    CubicFunctor<T> cFunc(c);
    residual[0] = m_z - cFunc(static_cast<T>(m_x), static_cast<T>(m_y));
    return true;
  }

private:
  const double m_x, m_y, m_z;
};

int main(int argc, char **argv)
{
  std::cout << __FILE__ << std::endl;

  // 1. Setting Up Sample Points.
  double minx = -5.0, maxx = 5.0, resox = 0.5;
  double miny = -5.0, maxy = 5.0, resoy = 0.5;
  int num_x = (maxx - minx) / resox + 1;
  int num_y = (maxy - miny) / resoy + 1;
  // Determine Coefficient
  std::vector<double> coeff{-0.2, 1.0, 1.0 - 0.2, 0.5, 1.0,
                            0.5, -1.0, -1.0, 1.0};

  // 2. Problem Formulation
  vector<vector<double>> x_elems(num_y), y_elems(num_y), z_elems(num_y),
      z_n_elems(num_y), z_est_elems(num_y);
  {
    // Create X Vector
    transform(x_elems.begin(), x_elems.end(), x_elems.begin(),
              [num_x, minx, resox](const vector<double> &v) {
                vector<double> vnew(num_x, 0.0);
                iota(vnew.begin(), vnew.end(), 0.0);
                transform(
                    vnew.begin(), vnew.end(), vnew.begin(),
                    [minx, resox](double val) { return val * resox + minx; });
                return vnew;
              });

    // Create Y Vector
    transform(y_elems.begin(), y_elems.end(), y_elems.begin(),
              [num_x, miny, resoy](const vector<double> &v) {
                static int cnt = 0;
                double val = resoy * cnt + miny;
                cnt++;
                vector<double> vnew(num_x, val);
                return vnew;
              });

    // Create Z Vector
    CubicFunctor<double> cFunc(coeff);
    for (int j = 0; j < num_y; j++)
    {
      for (int i = 0; i < num_x; i++)
      {
        double x = x_elems[j][i];
        double y = y_elems[j][i];
        z_elems[j].push_back(cFunc(x, y));
      }
    }

    z_n_elems = z_elems;
    NoiseAddingFunctor<double> nFunc(0.0, 10.0);
    for_each(z_n_elems.begin(), z_n_elems.end(), [&nFunc](vector<double> &v) {
      transform(v.begin(), v.end(), v.begin(), nFunc);
    });
  }

  // 3. Ceres Estimation
  vector<double> co(10, 0.0);
  {
    ceres::Problem problem;
    for (size_t j = 0; j < y_elems.size(); j++)
    {
      for (size_t i = 0; i < x_elems.size(); i++)
      {
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<CubicCostFunctor, 1, 10>(
                new CubicCostFunctor(x_elems[j][i], y_elems[j][i],
                                     z_n_elems[j][i]));
        problem.AddResidualBlock(cost_function, nullptr, co.data());
      }
    }

    // Run the solver.
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Status Report!
    std::cout << summary.BriefReport() << std::endl;

    std::cout << "Original Parameters" << std::endl;
    {
      std::ostream_iterator<double> out(std::cout, ", ");
      copy(coeff.begin(), coeff.end(), out);
      std::cout << std::endl
                << std::flush;
    }

    std::cout << "Estimated Parameters" << std::endl;
    {
      std::ostream_iterator<double> out(std::cout, ", ");
      copy(co.begin(), co.end(), out);
      std::cout << std::endl
                << std::flush;
    }

    // Create Estimated Z Vector
    CubicFunctor<double> cFuncEst(co);
    for (int j = 0; j < num_y; j++)
    {
      for (int i = 0; i < num_x; i++)
      {
        double x = x_elems[j][i];
        double y = y_elems[j][i];
        z_est_elems[j].push_back(cFuncEst(x, y));
      }
    }
  }

  // 4. Plot Result
  {
    plt::plot_surface(x_elems, y_elems, z_elems);
    plt::title("Original Surface");

    plt::plot_surface(x_elems, y_elems, z_n_elems);
    plt::title("Noised Surface");

    plt::plot_surface(x_elems, y_elems, z_est_elems);
    plt::title("Estimated Surface");

    plt::show();
  }
  return 0;
}
