

#ifndef CERES_EXAMPLES_POSE_GRAPH_2D_TYPES_H_
#define CERES_EXAMPLES_POSE_GRAPH_2D_TYPES_H_
#include "Eigen/Core"
#include "normalize_angle.h"
#include <fstream>

namespace ceres
{
namespace examples
{
// The state for each vertex in the pose graph.
struct Pose2d
{
  double x;
  double y;
  double yaw_radians;
  // The name of the data type in the g2o file format.
  static std::string name() { return "VERTEX_SE2"; }
};
std::istream &operator>>(std::istream &input, Pose2d &pose)
{
  input >> pose.x >> pose.y >> pose.yaw_radians;
  // Normalize the angle between -pi to pi.
  pose.yaw_radians = NormalizeAngle(pose.yaw_radians);
  return input;
}
// 構造体 Constraint は２つの頂点の位置の相対関係をほじすること．
struct Constraint2d
{
  int id_begin;
  int id_end;
  double x;
  double y;
  double yaw_radians;
  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, and yaw.
  Eigen::Matrix3d information;
  // The name of the data type in the g2o file format.
  static std::string name() { return "EDGE_SE2"; }
};
std::istream &operator>>(std::istream &input, Constraint2d &constraint)
{
  input >> constraint.id_begin >> constraint.id_end >> constraint.x >>
      constraint.y >> constraint.yaw_radians >> constraint.information(0, 0) >>
      constraint.information(0, 1) >> constraint.information(0, 2) >>
      constraint.information(1, 1) >> constraint.information(1, 2) >>
      constraint.information(2, 2);
  // Set the lower triangular part of the information matrix.
  constraint.information(1, 0) = constraint.information(0, 1);
  constraint.information(2, 0) = constraint.information(0, 2);
  constraint.information(2, 1) = constraint.information(1, 2);
  // Normalize the angle between -pi to pi.
  constraint.yaw_radians = NormalizeAngle(constraint.yaw_radians);
  return input;
}
} // namespace examples
} // namespace ceres
#endif // CERES_EXAMPLES_POSE_GRAPH_2D_TYPES_H_
