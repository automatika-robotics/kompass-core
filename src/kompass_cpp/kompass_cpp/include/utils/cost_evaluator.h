#pragma once

#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/transformation.h"
#include <array>
#include <cmath>
#include <memory>
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

      addParameter("smoothness_weight",
                   Parameter(1.0, 0.0, 1000.0,
                             "Weight of the cost for the non-smoothness of the "
                             "trajectory sample"));
      addParameter(
          "jerk_weight",
          Parameter(1.0, 0.0, 1000.0,
                    "Weight of the cost for the trajectory sample jerk"));
    }
  };

  /**
   * @brief Construct a new CostEvaluator.
   *
   * @param costsWeights              Per-cost weights (path, goal,
   *                                  obstacles, smoothness, jerk).
   * @param ctrLimits                 Robot control limits; acceleration
   *                                  limits are used to normalize smoothness
   *                                  and jerk costs.
   * @param maxNumTrajectories        Upper bound on trajectories per batch
   *                                  (sizes GPU device buffers at init).
   * @param numPointsPerTrajectory    Upper bound on points per trajectory
   *                                  (sizes GPU device buffers at init;
   *                                  actual per-batch count may be smaller
   *                                  and is read from the incoming samples).
   * @param maxRefPathSegmentSize     Expected tracked-segment size; GPU
   *                                  buffers grow on demand if exceeded.
   */
  CostEvaluator(TrajectoryCostsWeights &costsWeights,
                ControlLimitsParams ctrLimits, size_t maxNumTrajectories,
                size_t numPointsPerTrajectory, size_t maxRefPathSegmentSize);

  /**
   * @brief Construct a new CostEvaluator with a sensor-to-body transform.
   *
   * Obstacle points passed to setPointScan are interpreted in the sensor
   * frame and transformed into the body frame using this pose.
   *
   * @param costsWeights              See primary constructor.
   * @param sensor_position_body      Sensor position expressed in the body
   *                                  frame.
   * @param sensor_rotation_body      Sensor orientation (quaternion)
   *                                  expressed in the body frame.
   * @param ctrLimits                 See primary constructor.
   * @param maxNumTrajectories        See primary constructor.
   * @param numPointsPerTrajectory    See primary constructor.
   * @param maxRefPathSegmentSize     See primary constructor.
   */
  CostEvaluator(TrajectoryCostsWeights &costsWeights,
                const Eigen::Vector3f &sensor_position_body,
                const Eigen::Quaternionf &sensor_rotation_body,
                ControlLimitsParams ctrLimits, size_t maxNumTrajectories,
                size_t numPointsPerTrajectory, size_t maxRefPathSegmentSize);

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
      std::function<float(const Trajectory2D &, const Path::Path &)>;

  /**
   * @brief CustomTrajectoryCost is defined by a CustomCostFunction to evaluate
   * the cost and a weight assigned to that cost in the overall evaluation
   *
   */
  struct CustomTrajectoryCost {
    double weight;
    CustomCostFunction evaluator_;

    CustomTrajectoryCost(double weight, CustomCostFunction evaluator)
        : weight(weight), evaluator_(evaluator) {};
  };

  /**
   * @brief Evaluate every trajectory in `trajs` against `reference_path` and
   * return the lowest-cost one, with its total weighted cost.
   *
   * Applies each enabled cost (path, goal, obstacles, smoothness, jerk, plus
   * any custom costs) to every trajectory and returns the argmin. The tracked
   * segment is the window of the reference path the path- and goal-cost
   * kernels search over; callers should size it to at least cover the
   * trajectory's reach (prediction_horizon * maxVel) so longitudinal
   * overshoot isn't mis-read as lateral error.
   *
   * @param trajs             Batch of candidate trajectories.
   * @param reference_path    Full global reference path (interpolated; used
   *                          for prefix arc-length lookups in goal cost).
   * @param tracked_segment   Active View into reference_path.
   * @return TrajSearchResult Winning trajectory, a boolean "found" flag, and
   *                          the winning total cost.
   */
  TrajSearchResult
  getMinTrajectoryCost(const std::unique_ptr<TrajectorySamples2D> &trajs,
                       const Path::Path *reference_path,
                       const Path::Path::View &tracked_segment);

  /**
   * @brief Adds a new custom cost to be used in the trajectory evaluation
   *
   * @param weight
   * @param custom_cost_function
   */
  void addCustomCost(double weight, CustomCostFunction custom_cost_function) {
    auto newCost =
        std::make_unique<CustomTrajectoryCost>(weight, custom_cost_function);
    customTrajCostsPtrs_.push_back(std::move(newCost));
  };

  /**
   * @brief Load obstacle points from a polar LaserScan.
   *
   * Converts polar (range, angle) to Cartesian in the sensor frame, then
   * transforms into the body frame using the sensor-to-body pose and the
   * current robot pose. Results are used by the obstacles cost kernel.
   *
   * @param scan                              Polar scan.
   * @param current_state                     Current robot pose (world).
   * @param max_sensor_range                  Sensor's max valid range [m].
   * @param max_obstacle_cost_range_multiple  Divisor applied to
   *                                          max_sensor_range to produce
   *                                          maxObstaclesDist — the
   *                                          distance at which the
   *                                          obstacles cost reaches zero.
   *                                          Default 3.0 → cost vanishes
   *                                          at range/3.
   */
  void setPointScan(const LaserScan &scan, const Path::State &current_state,
                    const float max_sensor_range,
                    const float max_obstacle_cost_range_multiple = 3.0) {
    obstaclePointsX.clear();
    obstaclePointsY.clear();
    maxObstaclesDist = max_sensor_range / max_obstacle_cost_range_multiple;
    Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

    for (size_t i = 0; i < scan.ranges.size(); i++) {
      // Convert polar to Cartesian (assuming scan data in the XY plane)
      double point_x = scan.ranges[i] * std::cos(scan.angles[i]);
      double point_y = scan.ranges[i] * std::sin(scan.angles[i]);

      Eigen::Vector3f pose_trans =
          transformPosition(Eigen::Vector3f(point_x, point_y, 0.0),
                            sensor_tf_body_ * body_tf_world_);
      obstaclePointsX.emplace_back(pose_trans[0]);
      obstaclePointsY.emplace_back(pose_trans[1]);
    }
  };

  /**
   * @brief Load obstacle points from a Cartesian point cloud.
   *
   * Point cloud is assumed to be in the sensor frame; points are transformed
   * into the body frame using the sensor-to-body pose and the current robot
   * pose.
   *
   * @param cloud                             Point cloud in the sensor frame.
   * @param current_state                     Current robot pose (world).
   * @param max_sensor_range                  Sensor's max valid range [m].
   * @param max_obstacle_cost_range_multiple  See the LaserScan overload.
   */
  void setPointScan(const std::vector<Path::Point> &cloud,
                    const Path::State &current_state,
                    const float max_sensor_range,
                    const float max_obstacle_cost_range_multiple = 3.0) {
    obstaclePointsX.clear();
    obstaclePointsY.clear();
    maxObstaclesDist = max_sensor_range / max_obstacle_cost_range_multiple;
    Eigen::Isometry3f body_tf_world_ = getTransformation(current_state);

    for (auto &point : cloud) {
      Eigen::Vector3f pose_trans =
          transformPosition(Eigen::Vector3f(point.x(), point.y(), point.z()),
                            sensor_tf_body_ * body_tf_world_);
      obstaclePointsX.emplace_back(pose_trans[0]);
      obstaclePointsY.emplace_back(pose_trans[1]);
    }
  };

  /**
   * @brief Helper method to update the weights of the trajectory costs from
   * config
   *
   * @param costsWeights
   */
  void updateCostWeights(TrajectoryCostsWeights &costsWeights);

protected:
  // Protected member variables
  ControlType ctrType;
  std::array<float, 3> accLimits_;

  // Vector of pointers to the trajectory costs
  std::vector<std::unique_ptr<CustomTrajectoryCost>> customTrajCostsPtrs_;

private:
  std::unique_ptr<TrajectoryCostsWeights> costWeights;
  std::vector<float> obstaclePointsX;
  std::vector<float> obstaclePointsY;
  float
      maxObstaclesDist; // Distance at the maximum cost = max_robot_local_range
                        // / maxObstacleCostToRangeMultiple

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot

#ifdef GPU
  size_t numTrajectories_;
  size_t numPointsPerTrajectory_;
  size_t maxRefPathSegmentSize_;
  size_t maxObstaclePoints_; // updated at runtime
  size_t maxWGSize_;         // initialized from accelerator cpabilities
  float *m_devicePtrPathsX = nullptr;
  float *m_devicePtrPathsY = nullptr;
  float *m_devicePtrVelocitiesVx = nullptr;
  float *m_devicePtrVelocitiesVy = nullptr;
  float *m_devicePtrVelocitiesOmega = nullptr;
  float *m_devicePtrCosts = nullptr;
  float *m_devicePtrTrackedSegmentX = nullptr;
  float *m_devicePtrTrackedSegmentY = nullptr;
  float *m_devicePtrTrackedSegmentAccLengths =
      nullptr; // absolute prefix arc lengths for tracked segment points
  float *m_devicePtrObstaclesX = nullptr;
  float *m_devicePtrObstaclesY = nullptr;
  float *m_devicePtrTempCosts = nullptr;
  LowestCost *m_minCost;
  sycl::queue m_q;
  void initializeGPUMemory();
  /**
   * @brief Trajectory cost based on average cross-track error to the tracked
   * reference segment, plus normalized end-point distance.
   *
   * One workgroup per trajectory; each thread reduces over a stride-looped
   * set of trajectory points and finds the closest tracked-segment point per
   * trajectory point (tiled through local memory). Assumes tracked-segment
   * X/Y have already been uploaded to device memory by getMinTrajectoryCost.
   *
   * @param trajs_size                Number of trajectories in the batch.
   * @param tracked_segment_size      Number of points in the tracked segment.
   * @param tracked_segment_length    Arc length of the tracked segment [m]
   *                                  (used to normalize end-point distance).
   * @param cost_weight               Weight applied to the final cost before
   *                                  atomic-add into the per-trajectory
   *                                  cost accumulator.
   */
  sycl::event pathCostFunc(const size_t trajs_size,
                           const size_t tracked_segment_size,
                           const float tracked_segment_length,
                           const double cost_weight);

  /**
   * @brief Trajectory cost based on remaining arc-length along the reference
   * path from the trajectory endpoint to the goal.
   *
   * One workgroup per trajectory: each thread scans a stride-looped slice of
   * tracked-segment points for the one closest to the trajectory's endpoint;
   * a local-memory tree reduction then picks the workgroup-wide minimum.
   * The uploaded acc-length slice holds absolute prefix values on the full
   * reference path, so no segment-to-absolute index conversion is needed.
   *
   * Cost = arc_remaining / ref_path_length
   *      + sqrt(min_dist_sq) / ref_path_length        (tie-breaker)
   *
   * The base term is in [0, 1]. The tie-breaker prefers trajectories closer
   * to the tracked segment; it can push the total above 1 when the
   * trajectory endpoint is farther from the segment than the reference path
   * is long (rare, but not impossible on short paths with diverging samples).
   *
   * @param trajs_size            Number of trajectories in the batch.
   * @param tracked_segment_size  Number of points in the tracked segment.
   * @param ref_path_length       Total arc length of the full reference
   *                              path [m].
   * @param cost_weight           Weight applied before atomic-add into the
   *                              per-trajectory cost accumulator.
   */
  sycl::event goalCostFunc(const size_t trajs_size,
                           const size_t tracked_segment_size,
                           const float ref_path_length,
                           const double cost_weight);

  /**
   * @brief Trajectory smoothness cost: sum of squared first differences of
   * velocity, normalized per component by the corresponding acceleration
   * limit (cached as accLimits_ at construction).
   *
   * @param trajs_size    Number of trajectories in the batch.
   * @param cost_weight   Weight applied before atomic-add.
   */
  sycl::event smoothnessCostFunc(const size_t trajs_size,
                                 const double cost_weight);

  /**
   * @brief Trajectory jerk cost: sum of squared second differences of
   * velocity, normalized per component by the corresponding acceleration
   * limit (cached as accLimits_ at construction).
   *
   * @param trajs_size    Number of trajectories in the batch.
   * @param cost_weight   Weight applied before atomic-add.
   */
  sycl::event jerkCostFunc(const size_t trajs_size, const double cost_weight);

  /**
   * @brief Trajectory cost based on minimum distance to any obstacle point.
   *
   * Obstacle points are expected to already have been uploaded to device
   * memory by getMinTrajectoryCost (after setPointScan). Cost decays
   * linearly from 1 (obstacle touching a trajectory point) to 0 at
   * maxObstaclesDist.
   *
   * @param trajs_size    Number of trajectories in the batch.
   * @param cost_weight   Weight applied before atomic-add.
   */
  sycl::event obstaclesDistCostFunc(const size_t trajs_size,
                                    const double cost_weight);
#else
  // Built-in functions for cost evaluation
  /**
   * @brief Average cross-track error from every trajectory point to the
   * tracked reference segment, plus an end-point distance term.
   *
   * @param trajectory                Trajectory under evaluation.
   * @param tracked_segment           Active View into the reference path.
   * @param tracked_segment_length    Arc length of the tracked segment [m]
   *                                  (used to normalize end-point distance).
   * @return float                    Cost in meters (avg cross-track) with
   *                                  a normalized end-point term added.
   */
  float pathCostFunc(const Trajectory2D &trajectory,
                     const Path::Path::View &tracked_segment,
                     const float tracked_segment_length);

  /**
   * @brief Remaining arc-length along the reference path from the trajectory
   * endpoint to the goal. The closest point search runs over the tracked
   * segment; the segment-local index is converted to an absolute reference-
   * path index so the prefix arc length is looked up on the full path.
   *
   * Cost = arc_remaining / ref_path_length
   *      + sqrt(min_dist_sq) / ref_path_length        (tie-breaker)
   *
   * The base term is in [0, 1]. The tie-breaker prefers trajectories closer
   * to the tracked segment; it can push the total above 1 when the
   * trajectory endpoint is farther from the segment than the reference path
   * is long.
   *
   * @param trajectory        Trajectory under evaluation.
   * @param reference_path    Full reference path (for prefix arc-lengths).
   * @param tracked_segment   Active View over reference_path.
   * @param ref_path_length   Total arc length of the reference path [m].
   * @return float            arc_remaining/ref_len + tie-breaker.
   */
  float goalCostFunc(const Trajectory2D &trajectory,
                     const Path::Path *reference_path,
                     const Path::Path::View &tracked_segment,
                     const float ref_path_length);

  /**
   * @brief Trajectory cost based on minimum distance to any obstacle point
   * loaded via setPointScan. Linearly decays from 1 (obstacle touches a
   * trajectory point) to 0 at maxObstaclesDist.
   *
   * @param trajectory    Trajectory under evaluation.
   * @return float        Cost in [0, 1].
   */
  float obstaclesDistCostFunc(const Trajectory2D &trajectory);

  /**
   * @brief Trajectory smoothness cost: sum of squared first differences of
   * velocity, normalized per component by the corresponding acceleration
   * limit (cached as accLimits_ at construction).
   *
   * @param trajectory    Trajectory under evaluation.
   * @return float        Sum of (Δv)^2 / accLimit, averaged per component.
   */
  float smoothnessCostFunc(const Trajectory2D &trajectory);

  /**
   * @brief Trajectory jerk cost: sum of squared second differences of
   * velocity, normalized per component by the corresponding acceleration
   * limit (cached as accLimits_ at construction).
   *
   * @param trajectory    Trajectory under evaluation.
   * @return float        Sum of (Δ²v)^2 / accLimit, averaged per component.
   */
  float jerkCostFunc(const Trajectory2D &trajectory);
#endif //! GPU
};
}; // namespace Control
} // namespace Kompass
