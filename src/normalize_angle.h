
#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_NORMALIZE_ANGLE_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_NORMALIZE_ANGLE_H_
#include <cmath>
#include "ceres/ceres.h"
namespace ceres
{
namespace examples
{
// Normalizes the angle in radians between [-pi and pi).
template <typename T>
inline T NormalizeAngle(const T &angle_radians)
{
  // Use ceres::floor because it is specialized for double and Jet types.
  T two_pi(2.0 * M_PI);
  return angle_radians -
         two_pi * ceres::floor((angle_radians + T(M_PI)) / two_pi);
}
} // namespace examples
} // namespace ceres
#endif // CERES_EXAMPLES_POSE_GRAPH_2D_NORMALIZE_ANGLE_H_