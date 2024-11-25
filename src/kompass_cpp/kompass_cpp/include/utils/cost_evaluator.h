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
   * @brief Trajectory cost functions input arguments options:
   * The trajectory under evaluation,
   * or the trajectory and a path which can be the reference path (global path)
   * or a segment of it for example, or the trajectory and a state (which can be
   * the state of another robot, object etc.), or the trajectory and a point, or
   * the trajectory and a double (which can be time value, distance value,
   * etc.),
   */
  using CostFunctionArguments = std::variant<
      std::pair<const Trajectory, const Path::Path>,
      std::pair<const Trajectory, const Path::State>,
      std::pair<const Trajectory, const Path::Point>,
      std::pair<const Trajectory, const double>,
      std::tuple<const Trajectory, const Path::Path, const ControlType>,
      std::tuple<const Trajectory, const Path::Path, const double>,
      std::pair<const Trajectory, const std::vector<double>>,
      std::pair<const Trajectory, const std::vector<Point3D>>,
      std::pair<const Trajectory, const std::array<double, 3>>>;

  /**
   * @brief Function signature for cost functions
   *
   */
  using CostFunction = std::function<double(CostFunctionArguments)>;

  /**
   * @brief Function signature for any custom user defined cost function
   *
   */
  using CustomCostFunction =
      std::function<double(const Trajectory &, const Path::Path &)>;

  /**
   * @brief TrajectoryCost is defined by a CostFunction to evaluate the cost and
   * a weight assigned to that cost in the overall evaluation
   *
   */
  struct TrajectoryCost {
    double weight;
    CostFunction evaluator_;

    TrajectoryCost(double weight, CostFunction evaluator)
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
  double getTrajectoryCost(const Trajectory &traj,
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
   * @brief Helper function to wrap a trajectory cost function by parsing
   * CostFunctionArguments to the function
   *
   * @tparam Func
   * @tparam Arg
   * @param func
   * @return CostFunction
   */
  template <typename Func, typename Arg>
  CostFunction costFunctionWrapper(Func func) {
    return [func](CostFunctionArguments args) -> double {
      return std::visit(
          [func](auto &&arg) -> double {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, Arg>) {
              return std::apply(func, arg);
            } else {
              throw std::invalid_argument("Unexpected argument type");
            }
          },
          args);
    };
  }

  void setPointScan(const LaserScan &scan, const Path::State &curren_state);

  void setPointScan(const std::vector<Point3D> &cloud,
                    const Path::State &curren_state);

protected:
  // Protected member variables
  ControlType ctrType;
  ControlLimitsParams ctrlimits;
  CollisionChecker *collChecker;

  // Built-in Costs
  TrajectoryCost *referencePathDistCost = new TrajectoryCost(
      1.0, costFunctionWrapper<decltype(pathCostFunc),
                               std::tuple<const Trajectory, const Path::Path,
                                          const ControlType>>(pathCostFunc));

  TrajectoryCost *goalPointDistCost = new TrajectoryCost(
      1.0, costFunctionWrapper<decltype(goalCostFunc),
                               std::pair<const Trajectory, const Path::Path>>(
               goalCostFunc));

  TrajectoryCost *obstaclesDistCost = new TrajectoryCost(
      1.0, costFunctionWrapper<decltype(obstaclesDistCostFunc),
                               std::pair<const Trajectory, const Path::Path>>(
               obstaclesDistCostFunc));

  TrajectoryCost *smoothnessCost = new TrajectoryCost(
      1.0, costFunctionWrapper<
               decltype(smoothnessCostFunc),
               std::pair<const Trajectory, const std::array<double, 3>>>(
               smoothnessCostFunc));

  TrajectoryCost *jerkCost = new TrajectoryCost(
      1.0, costFunctionWrapper<
               decltype(jerkCostFunc),
               std::pair<const Trajectory, const std::array<double, 3>>>(
               jerkCostFunc));

  // Vector of pointers to the trajectory costs
  std::vector<TrajectoryCost *> customTrajCostsPtrs_;

private:
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
  void updateCostWeights(TrajectoryCostsWeights costsWeights);

  // Built-in functions for cost evaluation
  /**
   * @brief Trajectory cost based on the distance to a given reference path
   *
   * @param trajectory
   * @param reference_path
   * @return double
   */
  static double pathCostFunc(const Trajectory &trajectory,
                             const Path::Path &reference_path,
                             const ControlType controlType);

  /**
   * @brief Trajectory cost based on the distance to the end (goal) of a given
   * reference path
   *
   * @param trajectory
   * @param reference_path
   * @return double
   */
  static double goalCostFunc(const Trajectory &trajectory,
                             const Path::Path &reference_path);

  static double obstaclesDistCostFunc(const Trajectory &trajectory,
                                      const Path::Path &obstaclePoints);

  /**
   * @brief Trajectory cost based on the smoothness along the trajectory
   *
   * @param trajectory
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  static double smoothnessCostFunc(const Trajectory &trajectory,
                                   const std::array<double, 3> accLimits);

  /**
   * @brief Trajectory cost based on the jerk along the trajectory
   *
   * @param trajectory
   * @param accLimits     Robot acceleration limits [max acceleration on
   * x-direction, max on y-direction, max angular acceleration]
   * @return double
   */
  static double jerkCostFunc(const Trajectory &trajectory,
                             const std::array<double, 3> accLimits);
};
}; // namespace Control
} // namespace Kompass
