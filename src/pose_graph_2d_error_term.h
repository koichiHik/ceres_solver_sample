
#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_
#include "Eigen/Core"
namespace ceres {
namespace examples {
template <typename T> Eigen::Matrix<T, 2, 2> RotationMatrix2D(T yaw_radians) {
  const T cos_yaw = ceres::cos(yaw_radians);
  const T sin_yaw = ceres::sin(yaw_radians);
  Eigen::Matrix<T, 2, 2> rotation;
  rotation << cos_yaw, -sin_yaw, sin_yaw, cos_yaw;
  return rotation;
}
// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement.
//
// residual =  information^{1/2} * [  r_a^T * (p_b - p_a) - \hat{p_ab}   ]
//                                 [ Normalize(yaw_b - yaw_a - \hat{yaw_ab}) ]
//
// where r_a is the rotation matrix that rotates a vector represented in frame A
// into the global frame, and Normalize(*) ensures the angles are in the range
// [-pi, pi).
class PoseGraph2dErrorTerm {
public:
  PoseGraph2dErrorTerm(double x_ab, double y_ab, double yaw_ab_radians,
                       const Eigen::Matrix3d &sqrt_information)
      : p_ab_(x_ab, y_ab), yaw_ab_radians_(yaw_ab_radians),
        sqrt_information_(sqrt_information) {}
  template <typename T>
  // 関数演算子に渡ってくる値は，各ノードの世界座標．
  bool operator()(const T *const x_a, const T *const y_a, const T *const yaw_a,
                  const T *const x_b, const T *const y_b, const T *const yaw_b,
                  T *residuals_ptr) const {
    const Eigen::Matrix<T, 2, 1> p_a(*x_a, *y_a);
    const Eigen::Matrix<T, 2, 1> p_b(*x_b, *y_b);

    // Residualは3次元．
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_map(residuals_ptr);

    // Residualの計算ステップ．
    residuals_map.template head<2>() =
        RotationMatrix2D(*yaw_a).transpose() * (p_b - p_a) - p_ab_.cast<T>();
    residuals_map(2) = ceres::examples::NormalizeAngle(
        (*yaw_b - *yaw_a) - static_cast<T>(yaw_ab_radians_));

    residuals_map = sqrt_information_.template cast<T>() * residuals_map;
    return true;
  }

  // 2つのノード間の相対位置関係を保持する．
  static ceres::CostFunction *Create(double x_ab, double y_ab,
                                     double yaw_ab_radians,
                                     const Eigen::Matrix3d &sqrt_information) {
    // Residualは3次元で，後は各ノードの座標要素３つ x ２の合計６つ．
    // 初期値としてわかっている２つのノードの相対関係を使って，Functor
    // を初期化．
    return (new ceres::AutoDiffCostFunction<PoseGraph2dErrorTerm, 3, 1, 1, 1, 1,
                                            1, 1>(new PoseGraph2dErrorTerm(
        x_ab, y_ab, yaw_ab_radians, sqrt_information)));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  // The position of B relative to A in the A frame.
  const Eigen::Vector2d p_ab_;
  // The orientation of frame B relative to frame A.
  const double yaw_ab_radians_;
  // The inverse square root of the measurement covariance matrix.
  const Eigen::Matrix3d sqrt_information_;
};
} // namespace examples
} // namespace ceres
#endif // CERES_EXAMPLES_POSE_GRAPH_2D_POSE_GRAPH_2D_ERROR_TERM_H_