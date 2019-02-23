
#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_ANGLE_LOCAL_PARAMETERIZATION_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_ANGLE_LOCAL_PARAMETERIZATION_H_
#include "ceres/local_parameterization.h"
#include "normalize_angle.h"
namespace ceres
{
namespace examples
{
// Defines a local parameterization for updating the angle to be constrained in
// [-pi to pi).
class AngleLocalParameterization
{
public:
  template <typename T>
  bool operator()(const T *theta_radians, const T *delta_theta_radians,
                  T *theta_radians_plus_delta) const
  {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);
    return true;
  }
  static ceres::LocalParameterization *Create()
  {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};
} // namespace examples
} // namespace ceres
#endif // CERES_EXAMPLES_POSE_GRAPH_2D_ANGLE_LOCAL_PARAMETERIZATION_H_