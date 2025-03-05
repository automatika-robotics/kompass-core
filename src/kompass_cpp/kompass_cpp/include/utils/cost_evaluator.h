#pragma once

#include "collision_check.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <array>
#include <cmath>
#include <utility>
#include <variant>
#include <vector>

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
   * @brief Construct a new Trajectory Sampler object
   *
   * @param controlLimits
   * @param controlType
   * @param timeStep
   * @param timeHorizon
   * @param linearSampleStep
   * @param angularSampleStep
   * @param robotShapeType
   * @param robotDimensions
   * @param sensorPositionWRTbody
   * @param octreeRes
   */
  CostEvaluator(TrajectoryCostsWeights costWeights);

  CostEvaluator(TrajectoryCostsWeights costWeights,
                const std::array<float, 3> &sensor_position_body,
                const std::array<float, 4> &sensor_rotation_body);

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
  double getTrajectoryCost(const Trajectory2D &traj,
                           const Path::Path &reference_path,
                           const Path::Path &tracked_segment,
                           const size_t closest_segment_index);

  /**
   * @brief Adds a new custome cost to be used in the trajectory evaluation
   *
   * @param weight
   * @param custom_cost_function
   */
  void addCustomCost(double weight, CustomCostFunction custom_cost_function);

  /**
   * @brief Set the point scan with either lazerscan or vector of points
   *
   * @param scan / point cloud
   * @param current_state
   */
  void setPointScan(const LaserScan &scan, const Path::State &curren_state);

  void setPointScan(const std::vector<Path::Point> &cloud,
                    const Path::State &curren_state);

protected:
  // Protected member variables
  ControlType ctrType;
  ControlLimitsParams ctrlimits;
  CollisionChecker *collChecker;

  // Vector of pointers to the trajectory costs
  std::vector<CustomTrajectoryCost *> customTrajCostsPtrs_;

private:
  TrajectoryCostsWeights costWeights;
  std::vector<Path::Point> obstaclePoints;

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot

  /**
   * @brief Helper method to update the weights of the trajectory costs from
   * config
   *
   * @param costsWeights
   */
  void updateDefaultCostWeights(TrajectoryCostsWeights costsWeights);

  // Built-in functions for cost evaluation
  /**
   * @brief Trajectory cost based on the distance to a given reference path
   *
   * @param trajectory
   * @param reference_path
   * @return double
   */
  static double pathCostFunc(const Trajectory2D &trajectory,
                             const Path::Path &reference_path);

  /**
   * @brief Trajectory cost based on the distance to the end (goal) of a given
   * reference path
   *
   * @param trajectory
   * @param reference_path
   * @return double
   */
  static double goalCostFunc(const Trajectory2D &trajectory,
                             const Path::Path &reference_path);

  /**
   * @brief Trajectory cost based on the distance obstacles
   *
   * @param trajectory
   * @param obstaclePoints
   * @return double
   */
  static double
  obstaclesDistCostFunc(const Trajectory2D &trajectory,
                        const std::vector<Path::Point> &obstaclePoints);

  /**
   * @brief Trajectory cost based on the smoothness along the trajectory
   *
   * @param trajectory
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  static double smoothnessCostFunc(const Trajectory2D &trajectory,
                                   const std::array<double, 3> accLimits);

  /**
   * @brief Trajectory cost based on the jerk along the trajectory
   *
   * @param trajectory
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  static double jerkCostFunc(const Trajectory2D &trajectory,
                             const std::array<double, 3> accLimits);
};
}; // namespace Control
} // namespace Kompass
