

#include "angle_local_parameterization.h"
#include "ceres/ceres.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "pose_graph_2d_error_term.h"
#include "read_g2o.h"
#include "types.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
DEFINE_string(input, "", "The pose graph definition filename in g2o format.");
namespace ceres
{
namespace examples
{
// Constructs the nonlinear least squares optimization problem from the pose
// graph constraints.
void BuildOptimizationProblem(const std::vector<Constraint2d> &constraints,
                              std::map<int, Pose2d> *poses,
                              ceres::Problem *problem)
{
  CHECK(poses != NULL);
  CHECK(problem != NULL);
  if (constraints.empty())
  {
    LOG(INFO) << "No constraints, no problem to optimize.";
    return;
  }
  ceres::LossFunction *loss_function = NULL;
  ceres::LocalParameterization *angle_local_parameterization =
      AngleLocalParameterization::Create();

  // Pose Graph の要素でループを回す．
  for (std::vector<Constraint2d>::const_iterator constraints_iter =
           constraints.begin();
       constraints_iter != constraints.end(); ++constraints_iter)
  {

    // Constraint を構成する２つのノードを検索する．
    const Constraint2d &constraint = *constraints_iter;
    std::map<int, Pose2d>::iterator pose_begin_iter =
        poses->find(constraint.id_begin);
    CHECK(pose_begin_iter != poses->end())
        << "Pose with ID: " << constraint.id_begin << " not found.";
    std::map<int, Pose2d>::iterator pose_end_iter =
        poses->find(constraint.id_end);
    CHECK(pose_end_iter != poses->end())
        << "Pose with ID: " << constraint.id_end << " not found.";
    const Eigen::Matrix3d sqrt_information =
        constraint.information.llt().matrixL();

    // Constraint（このペア）に対する Cost Function を生成する．
    ceres::CostFunction *cost_function = PoseGraph2dErrorTerm::Create(
        constraint.x, constraint.y, constraint.yaw_radians, sqrt_information);

    // Cost Function と初期値を登録する．初期値はそれぞれのノードの座標値．
    problem->AddResidualBlock(
        cost_function, loss_function, &pose_begin_iter->second.x,
        &pose_begin_iter->second.y, &pose_begin_iter->second.yaw_radians,
        &pose_end_iter->second.x, &pose_end_iter->second.y,
        &pose_end_iter->second.yaw_radians);

    problem->SetParameterization(&pose_begin_iter->second.yaw_radians,
                                 angle_local_parameterization);

    problem->SetParameterization(&pose_end_iter->second.yaw_radians,
                                 angle_local_parameterization);
  }
  // The pose graph optimization problem has three DOFs that are not fully
  // constrained. This is typically referred to as gauge freedom. You can apply
  // a rigid body transformation to all the nodes and the optimization problem
  // will still have the exact same cost. The Levenberg-Marquardt algorithm has
  // internal damping which mitigate this issue, but it is better to properly
  // constrain the gauge freedom. This can be done by setting one of the poses
  // as constant so the optimizer cannot change it.
  // 座標基準選択の３自由度が残っているので，これを決める．
  std::map<int, Pose2d>::iterator pose_start_iter = poses->begin();
  CHECK(pose_start_iter != poses->end()) << "There are no poses.";
  problem->SetParameterBlockConstant(&pose_start_iter->second.x);
  problem->SetParameterBlockConstant(&pose_start_iter->second.y);
  problem->SetParameterBlockConstant(&pose_start_iter->second.yaw_radians);
}
// Returns true if the solve was successful.
bool SolveOptimizationProblem(ceres::Problem *problem)
{
  CHECK(problem != NULL);
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  std::cout << summary.FullReport() << '\n';
  return summary.IsSolutionUsable();
}
// Output the poses to the file with format: ID x y yaw_radians.
bool OutputPoses(const std::string &filename,
                 const std::map<int, Pose2d> &poses)
{
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile)
  {
    std::cerr << "Error opening the file: " << filename << '\n';
    return false;
  }
  for (std::map<int, Pose2d>::const_iterator poses_iter = poses.begin();
       poses_iter != poses.end(); ++poses_iter)
  {
    const std::map<int, Pose2d>::value_type &pair = *poses_iter;
    outfile << pair.first << " " << pair.second.x << " " << pair.second.y << ' '
            << pair.second.yaw_radians << '\n';
  }
  return true;
}
} // namespace examples
} // namespace ceres

// 最適化実行．
int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  //  CERES_GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(FLAGS_input != "") << "Need to specify the filename to read.";
  std::map<int, ceres::examples::Pose2d> poses;
  std::vector<ceres::examples::Constraint2d> constraints;
  CHECK(ceres::examples::ReadG2oFile(FLAGS_input, &poses, &constraints))
      << "Error reading the file: " << FLAGS_input;
  std::cout << "Number of poses: " << poses.size() << '\n';
  std::cout << "Number of constraints: " << constraints.size() << '\n';
  CHECK(ceres::examples::OutputPoses("poses_original.txt", poses))
      << "Error outputting to poses_original.txt";

  // 最適化問題の構築．
  ceres::Problem problem;
  ceres::examples::BuildOptimizationProblem(constraints, &poses, &problem);

  // ソルバ実行．
  CHECK(ceres::examples::SolveOptimizationProblem(&problem))
      << "The solve was not successful, exiting.";

  // 最終結果出力．
  CHECK(ceres::examples::OutputPoses("poses_optimized.txt", poses))
      << "Error outputting to poses_original.txt";
  return 0;
}