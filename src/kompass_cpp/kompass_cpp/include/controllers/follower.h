/**
 * *********************************************************
 *
 * @file follower.h
 * @author Based on the implementation of Alexander Buchegger
 * @brief
 * @version 0.1
 * @date 2024-06-20
 *
 * @copyright Copyright (c) 2024. ARTI - Autonomous Robot Technology GmbH. All
 * rights reserved.
 *
 */
#pragma once

#include "controller.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include <cmath>

namespace Kompass {
namespace Control {

class Follower : public Controller {
public:
  // Nested class for follower parameters
  class FollowerParameters : public Controller::ControllerParameters {
  public:

    FollowerParameters() : Controller::ControllerParameters() {
      addParameter("max_point_interpolation_distance",
                   Parameter(0.01, 0.0001,
                             1000.0)); // [m] distance used for path interpolation
      addParameter(
          "lookahead_distance",
          Parameter(1.0, 0.0,
                    1000.0)); // [m] Lookahead distance used to find the next
                            // point to reach (normally be same as wheelbase)
      addParameter(
          "goal_dist_tolerance",
          Parameter(0.1, 0.001, 1000.0)); // [m] Tolerance to consider the robot
                                        // reached the goal point
      addParameter(
          "path_segment_length",
          Parameter(1.0, 0.001,
                    1000.0)); // [m] Length of one segment of the path to follow
      addParameter(
          "goal_orientation_tolerance",
          Parameter(0.1, 0.001, 2 * M_PI)); // [rad] Tolerance to consider the robot
                                       // reached the goal point
      addParameter("loosing_goal_distance",
                   Parameter(0.1, 0.001, 1000.0)); // [m] If driving past the goal
                                                 // we stop after this distance
    }
  };

  /**
   * @brief Struct for tracked point "Target" information
   *
   */
  struct Target {
    size_t segment_index{0};
    double position_in_segment{0.0};
    Path::State movement = Path::State();
    bool reverse{false};
    double lookahead{0.0};
    double crosstrack_error{0.0};
    double heading_error{0.0};
  };

  // Constructor
  Follower();

  Follower(FollowerParameters config);

  // Destructor
  virtual ~Follower();

  /**
   * @brief Sets the global path to be followed
   * Sets the given path as a reference path and interpolates more points using
   * spline interpolation and segments the interpolated path
   *
   * @param path Global path to be followed
   */
  void setCurrentPath(const Path::Path &path);

  /**
   * @brief Checks if the currenState reached end of the path and returns a
   * boolean value
   *
   * @return true if goal is reached or missed
   * @return false is goal is still not reached
   */
  bool isGoalReached();

  /**
   * @brief Set the Interpolation type used for interpolating the follower's path
   *
   * @param type
   */
  void setInterpolationType(Path::InterpolationType type);

  bool isForwardSegment(const Path::Path &segment1,
                        const Path::Path &segment2) const;

  size_t getCurrentSegmentIndex();

  /**
   * @brief Gets information on the current path point getting tracked by the
   * follower
   *
   * @return Follower::Target
   */
  Target getTrackedTarget() const;

  /**
   * @brief Get the Linear Velocity Cmd X object
   *
   * @return double
   */
  double getLinearVelocityCmdX() const;

  /**
   * @brief Get the Linear Velocity Cmd Y object
   *
   * @return double
   */
  double getLinearVelocityCmdY() const;

  /**
   * @brief Get the Angular Velocity Cmd object
   *
   * @return double
   */
  double getAngularVelocityCmd() const;

  /**
   * @brief Get the Steering Angle Cmd object
   *
   * @return double
   */
  double getSteeringAngleCmd() const;

  /**
   * @brief Get the Path Length object
   *
   * @return const double
   */
  const double getPathLength() const;

  /**
   * @brief Checks if there is a path to follow
   *
   * @return true
   * @return false
   */
  const bool hasPath() const;

  const Path::Path getCurrentPath() const;

  /**
   * @brief Helper method to calculate the distance between a state and a point
   *
   * @param state
   * @param point
   * @return const double
   */
  const double calculateDistance(const Path::State &state,
                                 const Path::Point &point) const;

  /**
   * @brief Helper method to calculate the distance between two points
   *
   * @param point1
   * @param point2
   * @return const double
   */
  const double calculateDistance(const Path::Point &point1,
                                 const Path::Point &point2) const;

  /**
   * @brief Helper method to calculate the distance between two states
   *
   * @param state1
   * @param state2
   * @return const double
   */
  const double calculateDistance(const Path::State &state1,
                                 const Path::State &state2) const;

protected:
  Path::Path *currentPath = new Path::Path();
  Path::Path *refPath = new Path::Path();
  Path::PathPosition *closestPosition = new Path::PathPosition();
  double goal_dist_tolerance{0.0};
  double goal_orientation_tolerance{0.0};
  double loosing_goal_distance{0.0};
  bool rotate_in_place{false};
  double lookahead_distance{0.0};
  bool enable_reverse_driving{false};
  double path_segment_length{0.0};
  double maxDist{0.0};
  Path::InterpolationType interpolationType = Path::InterpolationType::LINEAR;

  FollowerParameters config;

  /**
   * @brief Finds closestPosition on the currentPath to the currentState
   * Performs a recursive search to first find the closest segment, then the
   * closest point of the segment
   *
   * @return Path::PathPosition
   */
  Path::PathPosition findClosestPathPoint();

  // bool isForwardMovement(const Path::State &tracked_position);

  /**
   * @brief Computes the path tracking target
   *
   */
  void determineTarget();

  bool path_processing_{false};
  Target *currentTrackedTarget_ = new Target();

  size_t current_segment_index_{0};
  double current_position_in_segment_{0.0};
  size_t max_segment_index_{0};

  double goal_distance_{std::numeric_limits<double>::max()};
  double goal_orientation_{std::numeric_limits<double>::max()};

  Control::Velocity latest_velocity_command_{0.0, 0.0, 0.0};

  bool reached_goal_{false};
  bool reached_yaw_{false};

private:

  /**
   * @brief Finds the index of the closest segment on the path to the
   * currentState between two given segment indices
   *
   * @param left Index of the left segment
   * @param right Index of the right segment
   * @return size_t   Closest segment index
   */
  size_t findClosestSegmentIndex(size_t left, size_t right);

  /**
   * @brief Finds the closest point to currentState on a given path segment
   *
   * @param segment_index     Path segment index
   * @return Path::PathPosition
   */
  Path::PathPosition findClosestPointOnSegment(size_t segment_index);

  /**
   * @brief Helper method to find the projection of a point onto a line segment
   * between two points
   *
   * @param a     Point start of the segment
   * @param b     Point send of the segment
   * @param segment_length        Segment length to be updated
   * @return Path::State          Projection state
   */
  Path::State projectPointOnSegment(const Path::Point &a, const Path::Point &b,
                                    double &segment_length);
};

} // namespace Control
} // namespace Kompass
