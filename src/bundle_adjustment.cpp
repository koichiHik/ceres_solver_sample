
#include <cmath>
#include <cstdio>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

// Google Log
#include "glog/logging.h"

class BALProblem {
public:
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  int num_observations() const { return num_observations_; }
  const double *observations() const { return observations_; }
  double *mutable_cameras() { return parameters_; }
  double *mutable_points() { return parameters_ + 9 * num_cameras_; }

  double *mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 9;
  }
  double *mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  bool LoadFile(const char *filename) {
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    }

    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    // 同一の点が異なるカメラ or 異なる視点から観測されている．
    // Observation に対応する Point Index, Camera Index がある．
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    // 最適化して求める変数の数は
    // カメラ台数 x ９自由度 + 点数 X 3自由度
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      // Camear Index and Point Index.
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);

      // Image Coords of Observations.
      FscanfOrDie(fptr, "%lf", observations_ + 2 * i);
      FscanfOrDie(fptr, "%lf", observations_ + 2 * i + 1);
    }

    // ここが何を読んでいるかわからん．．．
    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

    return true;
  }

private:
  template <typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int *point_index_;
  int *camera_index_;
  double *observations_;
  double *parameters_;
};

struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x_(observed_x), observed_y_(observed_y) {}

  template <typename T>
  bool operator()(const T *const camera, const T *const point,
                  T *residuals) const {
    // Angle Axis Rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // Translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T &l1 = camera[7];
    const T &l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);

    // Compute Final Projected Point Position.
    const T &focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // Reprojection Error.
    residuals[0] = predicted_x - observed_x_;
    residuals[1] = predicted_y - observed_y_;

    return true;
  }

  static ceres::CostFunction *Create(const double observed_x,
                                     const double observed_y) {
    return new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y));
  }

  // Observed Pixel Coords. This is fixed.
  double observed_x_;
  double observed_y_;
};

int main(int argc, char **argv) {
  std::cout << __FILE__ << " Started!" << std::endl;
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: Unable to open file " << argv[1] << std::endl;
  }

  const double *observations = bal_problem.observations();

  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(
        observations[2 * i + 0], observations[2 * i + 1]);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  return 0;
}