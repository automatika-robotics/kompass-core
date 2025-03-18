#pragma once

#include "collision_check.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/transformation.h"
#include <array>
#include <cmath>
#include <vector>
#ifdef GPU
#include <sycl/sycl.hpp>
#endif //! GPU

namespace Kompass {

namespace Control {
class CostEvaluator {
public:
  class TrajectoryCostsWeights : public Parameters {
  public:
    TrajectoryCostsWeights() : Parameters() {
      addParameter(
          "reference_path_distance_weight",
          Parameter(1.0, 0.0, 1000.0,
                    "Weight of the cost for the distance between a trajectory "
                    "sample and the reference global path"));
      addParameter(
          "goal_distance_weight",
          Parameter(1.0, 0.0, 1000.0,
                    "Weight of the cost for the distance between the end of a "
                    "trajectory sample and the end goal point"));
      addParameter(
          "obstacles_distance_weight",
          Parameter(1.0, 0.0, 1000.0,
                    "Weight of the cost for the distance between a trajectory "
                    "sample and the closest obstacle"));

      addParameter(
          "smoothness_weight",
          Parameter(
              1.0, 0.0, 1000.0,
              "Weight of the cost for the  trajectory sample non smoothness"));
      addParameter(
          "jerk_weight",
          Parameter(1.0, 0.0, 1000.0,
                    "Weight of the cost for the trajectory sample jerk"));
    }
  };

  /**
   * @brief Construct a new Cost evaluator object
   *
   * @param costWeights
   * @param controlType
   * @param timeStep
   * @param timeHorizon
   * @param maxLinearSample
   * @param maxAngularSample
   */
  CostEvaluator(TrajectoryCostsWeights costsWeights, ControlType controlType,
                ControlLimitsParams ctrLimits, double timeStep,
                double timeHorizon, size_t maxLinearSamples,
                size_t maxAngularSamples, size_t maxPathLength);
  CostEvaluator(TrajectoryCostsWeights costsWeights,
                const std::array<float, 3> &sensor_position_body,
                const std::array<float, 4> &sensor_rotation_body,
                ControlType controlType, ControlLimitsParams ctrLimits,
                double timeStep, double timeHorizon, size_t maxLinearSamples,
                size_t maxAngularSamples, size_t maxPathLength);

  /**
   * @brief Destroy the Trajectory Sampler object
   *
   */
  ~CostEvaluator();

  /**
   * @brief Function signature for any custom user defined cost function
   *
   */
  using CustomCostFunction =
      std::function<double(const Trajectory2D &, const Path::Path &)>;

  /**
   * @brief CustomTrajectoryCost is defined by a CustomCostFunction to evaluate
   * the cost and a weight assigned to that cost in the overall evaluation
   *
   */
  struct CustomTrajectoryCost {
    double weight;
    CustomCostFunction evaluator_;

    CustomTrajectoryCost(double weight, CustomCostFunction evaluator)
        : weight(weight), evaluator_(evaluator) {}
  };

  /**
   * @brief Get the Trajectory Cost by applying all the defined cost functions
   *
   * @param traj      The trajectory under evaluation
   * @param reference_path        The reference path (global path)
   * @param closest_segment_index     The segment of the closest segment from
   * the global path
   * @return double
   */
  TrajSearchResult getMinTrajectoryCost(const TrajectorySamples2D &trajs,
                                        const Path::Path &reference_path,
                                        const Path::Path &tracked_segment,
                                        const size_t closest_segment_index);

  /**
   * @brief Adds a new custome cost to be used in the trajectory evaluation
   *
   * @param weight
   * @param custom_cost_function
   */
  void addCustomCost(double weight, CustomCostFunction custom_cost_function) {
    CustomTrajectoryCost *newCost =
        new CustomTrajectoryCost(weight, custom_cost_function);
    customTrajCostsPtrs_.push_back(newCost);
  };

  /**
   * @brief Set the point scan with either lazerscan or vector of points
   *
   * @param scan / point cloud
   * @param current_state
   */
  void setPointScan(const LaserScan &scan, const Path::State &current_state) {
    obstaclePoints.clear();
    Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

    for (size_t i = 0; i < scan.ranges.size(); i++) {
      // Convert polar to Cartesian (assuming scan data in the XY plane)
      double point_x = scan.ranges[i] * std::cos(scan.angles[i]);
      double point_y = scan.ranges[i] * std::sin(scan.angles[i]);

      Eigen::Vector3f pose_trans =
          transformPosition(Eigen::Vector3f(point_x, point_y, 0.0),
                            sensor_tf_body_ * body_tf_world_);
      obstaclePoints.push_back({pose_trans[0], pose_trans[1]});
    }
  };

  void setPointScan(const std::vector<Path::Point> &cloud,
                    const Path::State &current_state) {
    obstaclePoints.clear();
    Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

    for (auto &point : cloud) {
      // TODO: fix
      Eigen::Vector3f pose_trans =
          transformPosition(Eigen::Vector3f(point.x(), point.y(), point.z()),
                            sensor_tf_body_ * body_tf_world_);
      obstaclePoints.push_back({pose_trans[0], pose_trans[1]});
    }
  };

  /**
   * @brief Helper method to update the weights of the trajectory costs from
   * config
   *
   * @param costsWeights
   */
  void updateDefaultCostWeights(TrajectoryCostsWeights costsWeights) {
    this->costWeights = costsWeights;
  };

protected:
  // Protected member variables
  ControlType ctrType;
  CollisionChecker *collChecker;
  std::array<double, 3> accLimits_;

  // Vector of pointers to the trajectory costs
  std::vector<CustomTrajectoryCost *> customTrajCostsPtrs_;

private:
  TrajectoryCostsWeights costWeights;
  std::vector<Path::Point> obstaclePoints;

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot

#ifdef GPU
  size_t numTrajectories_;
  size_t numPointsPerTrajectory_;
  double *m_devicePtrPathsX;
  double *m_devicePtrPathsY;
  double *m_devicePtrVelocitiesVx;
  double *m_devicePtrVelocitiesVy;
  double *m_devicePtrVelocitiesOmega;
  float *m_devicePtrReferencePathX;
  float *m_devicePtrReferencePathY;
  double *m_devicePtrCosts;
  double *m_devicePtrTempCosts;
  LowestCost *m_minCost;
  sycl::queue m_q;
  void initializeGPUMemory(TrajectoryCostsWeights costWeights, size_t maxPathLength);
  /**
   * @brief Trajectory cost based on the distance to a given reference path
   *
   * @param trajectories
   * @param reference_path
   * @return double
   */
  void pathCostFunc(const size_t trajs_size, const size_t ref_path_size,
                    const double cost_weight);

  /**
   * @brief Trajectory cost based on the distance to the end (goal) of a given
   * reference path
   *
   * @param trajectories
   * @param reference_path
   * @return double
   */
  void goalCostFunc(const size_t trajs_size, const size_t ref_path_size,
                    const double path_length, const double cost_weight);

  /**
   * @brief Trajectory cost based on the smoothness along the trajectory
   *
   * @param trajectories
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  void smoothnessCostFunc(const size_t trajs_size, const double cost_weight);

  /**
   * @brief Trajectory cost based on the jerk along the trajectory
   *
   * @param trajectories
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  void jerkCostFunc(const size_t trajs_size, const double cost_weight);

  /**
   * @brief Trajectory cost based on the distance obstacles
   *
   * @param trajectory
   * @param obstaclePoints
   * @return double
   */
  double obstaclesDistCostFunc(const Trajectory2D &trajectory,
                               const std::vector<Path::Point> &obstaclePoints);
#else
  // Built-in functions for cost evaluation
  /**
   * @brief Trajectory cost based on the distance to a given reference path
   *
   * @param trajectory
   * @param reference_path
   * @return double
   */
  double pathCostFunc(const Trajectory2D &trajectory,
                      const Path::Path &reference_path);

  /**
   * @brief Trajectory cost based on the distance to the end (goal) of a given
   * reference path
   *
   * @param trajectory
   * @param reference_path
   * @return double
   */
  double goalCostFunc(const Trajectory2D &trajectory,
                      const Path::Path &reference_path);

  /**
   * @brief Trajectory cost based on the distance obstacles
   *
   * @param trajectory
   * @param obstaclePoints
   * @return double
   */
  double obstaclesDistCostFunc(const Trajectory2D &trajectory,
                               const std::vector<Path::Point> &obstaclePoints);

  /**
   * @brief Trajectory cost based on the smoothness along the trajectory
   *
   * @param trajectory
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  double smoothnessCostFunc(const Trajectory2D &trajectory);

  /**
   * @brief Trajectory cost based on the jerk along the trajectory
   *
   * @param trajectory
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  double jerkCostFunc(const Trajectory2D &trajectory);
#endif //! GPU
};
}; // namespace Control
} // namespace Kompass
