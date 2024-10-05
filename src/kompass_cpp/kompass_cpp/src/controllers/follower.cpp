#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "controllers/follower.h"
#include "utils/angles.h"
#include "utils/logger.h"

using namespace std;

namespace Kompass {
namespace Control {

Follower::Follower() : Controller(), config() {
  // Get parameters from config
  lookahead_distance = config.getParameter<double>("lookahead_distance");
  enable_reverse_driving = config.getParameter<bool>("enable_reverse_driving");
  goal_dist_tolerance = config.getParameter<double>("goal_dist_tolerance");
  goal_orientation_tolerance =
      config.getParameter<double>("goal_orientation_tolerance");
  loosing_goal_distance = config.getParameter<double>("loosing_goal_distance");
  path_segment_length = config.getParameter<double>("path_segment_length");
  maxDist = config.getParameter<double>("max_point_interpolation_distance");
  // Set rotate_in_place based on the robot type
  if (ctrType == Control::ControlType::ACKERMANN) {
    rotate_in_place = false;
  } else {
    rotate_in_place = true;
  }
}

Follower::Follower(FollowerParameters config) : Controller() {
  this->config = config;
  lookahead_distance = config.getParameter<double>("lookahead_distance");
  enable_reverse_driving = config.getParameter<bool>("enable_reverse_driving");
  goal_dist_tolerance = config.getParameter<double>("goal_dist_tolerance");
  goal_orientation_tolerance =
      config.getParameter<double>("goal_orientation_tolerance");
  loosing_goal_distance = config.getParameter<double>("loosing_goal_distance");
  path_segment_length = config.getParameter<double>("path_segment_length");
  maxDist = config.getParameter<double>("max_point_interpolation_distance");
  // Set rotate_in_place based on the robot type
  if (ctrType == Control::ControlType::ACKERMANN) {
    rotate_in_place = false;
  } else {
    rotate_in_place = true;
  }
}

Follower::~Follower() {
  delete refPath;
  delete currentPath;
  delete closestPosition;
}

Follower::Target Follower::getTrackedTarget() const {
  return *currentTrackedTarget_;
}

const Path::Path Follower::getCurrentPath() const { return *currentPath; }

void Follower::setCurrentPath(const Path::Path &path) {
  // Delete old reference and current path before setting new values
  delete refPath;
  delete currentPath;

  currentPath = new Path::Path();
  refPath = new Path::Path();

  refPath->points = path.points;


  currentPath->points = refPath->points;

  currentPath->interpolate(maxDist, interpolationType);

  // Segment path
  currentPath->segment(path_segment_length);

  // Get max number of segments in the path
  max_segment_index_ = currentPath->getMaxNumSegments();

  path_processing_ = true;
  current_segment_index_ = 0;
  current_position_in_segment_ = 0.0;

  goal_distance_ = std::numeric_limits<double>::max();
  goal_orientation_ = currentPath->getEndOrientation();

  reached_goal_ = false;
  reached_yaw_ = false;

  return;
}

size_t Follower::getCurrentSegmentIndex() { return current_segment_index_; }

bool Follower::isGoalReached() {
  if (!path_processing_) {
    return reached_goal_;
  }

  const Path::Point goal_point = currentPath->getEnd();
  bool loosing_goal = false;

  const double current_goal_distance =
      std::hypot(currentState.x - goal_point.x, currentState.y - goal_point.y);

  bool end_reached = current_goal_distance <= goal_dist_tolerance;

  // check if we are loosing the goal - stop driving if this happens
  if ((current_segment_index_ + 1) >= max_segment_index_) {
    // At end of path
    // Check if moving closer to end
    if (current_goal_distance < goal_distance_) {
      goal_distance_ = current_goal_distance;
      loosing_goal = false;
    } else if (std::abs(current_goal_distance - goal_distance_) >
               loosing_goal_distance) {
      LOG_DEBUG("Already Reached the Goal, Ending Action\n");
      loosing_goal = true;
    }
  }

  if (end_reached || loosing_goal) {
    path_processing_ = false;
    reached_goal_ = true;
  }

  return reached_goal_;
}

void Follower::setInterpolationType(Path::InterpolationType type) {
  interpolationType = type;
}

const double Follower::calculateDistance(const Path::State &state,
                                         const Path::Point &point) const {
  return std::hypot(state.x - point.x, state.y - point.y);
}

const double Follower::calculateDistance(const Path::Point &point1,
                                         const Path::Point &point2) const {
  return std::hypot(point1.x - point2.x, point1.y - point2.y);
}

const double Follower::calculateDistance(const Path::State &state1,
                                         const Path::State &state2) const {
  return std::hypot(state1.x - state2.x, state1.y - state2.y);
}

// Method to find the closest point on the path to the current position
Path::PathPosition Follower::findClosestPathPoint() {
  current_segment_index_ = findClosestSegmentIndex(0, max_segment_index_);
  return findClosestPointOnSegment(current_segment_index_);
}

// Method to find the closest segment index using a binary search-like approach
size_t Follower::findClosestSegmentIndex(size_t left, size_t right) {
  if (left == right) {
    return left;
  }

  size_t mid = (left + right) / 2;

  // In case only two points are available
  if (mid == right || mid == left) {
    double left_distance =
        calculateDistance(currentState, currentPath->segments[left].getStart());
    double right_distance = calculateDistance(
        currentState, currentPath->segments[right].getStart());
    if (left_distance <= right_distance) {
      return left;
    } else {
      return right;
    }
  }

  double left_distance =
      calculateDistance(currentState, currentPath->segments[left].getStart());
  double right_distance =
      calculateDistance(currentState, currentPath->segments[right].getStart());

  if (left_distance <= right_distance) {
    return findClosestSegmentIndex(left, mid);
  } else {
    return findClosestSegmentIndex(mid, right);
  }
}

Path::State Follower::projectPointOnSegment(const Path::Point &a,
                                            const Path::Point &b,
                                            double &segment_length) {
  double ab_x = b.x - a.x;
  double ab_y = b.y - a.y;

  double segment_heading = std::atan2(
      ab_y,
      ab_x); // + M_PI_2; // PI/2 is added to measure angle to x axis not y
  segment_length = std::sqrt(ab_x * ab_x + ab_y + ab_y);

  return Path::State(a.x, a.y, segment_heading);
}

Path::PathPosition Follower::findClosestPointOnSegment(size_t segment_index) {

  const std::vector<Path::Point> &segment_points =
      currentPath->segments[segment_index].points;
  double min_distance = std::numeric_limits<double>::max();
  Path::State closest_point;
  double segment_position = 0.0;  // in [0, 1]
  size_t point_index = 0;
  // Get current segment start, end to calculate length and orientation
  Path::Point start = currentPath->segments[segment_index].getStart();
  Path::Point end = currentPath->segments[segment_index].getEnd();

  double segment_heading = std::atan2(end.y - start.y, end.x - start.x);

  for (size_t i = 0; i < segment_points.size(); ++i) {

    Path::Point projected_point = segment_points[i];
    double distance = calculateDistance(currentState, projected_point);

    if (distance < min_distance) {
      min_distance = distance;
      closest_point = {projected_point.x, projected_point.y, segment_heading};
      segment_position =
          static_cast<double>(i) / (segment_points.size() - 1 );
      point_index = i;
    }
  }

  Path::PathPosition closest_position;
  closest_position.index = point_index;
  closest_position.segment_index = segment_index;
  closest_position.segment_length = segment_position;
  closest_position.state = closest_point;
  closest_position.normal_distance = min_distance;

  // Compute parallel distance (signed lateral distance)
  double vec_x = currentState.x - closest_point.x;
  double vec_y = currentState.y - closest_point.y;

  // Calculate the direction vector based on the heading
  double cos_direction = std::cos(closest_point.yaw);
  double sin_direction = std::sin(closest_point.yaw);

  // crossProduct determines if the currentState is left or right to the closest
  // point
  double crossProduct = cos_direction * vec_y - sin_direction * vec_x;

  closest_position.parallel_distance =
      crossProduct > 0 ? min_distance : -min_distance;

  return closest_position;
}

void Follower::determineTarget() {
  // delete closestPosition;
  delete currentTrackedTarget_;

  currentTrackedTarget_ = new Target();
  // closestPosition = new Path::PathPosition();

  // closest position is never updated
  if (closestPosition->segment_length <= 0.0) {
    *closestPosition = findClosestPathPoint();
  }
  // If we reached end of segment or end of path -> Find new point
  else if (closestPosition->segment_index >= currentPath->points.size() - 1 ||
           closestPosition->segment_length >= 1.0) {
    *closestPosition = findClosestPathPoint();
  }
  // If end of segment is not reached -> only find closest point on segment
  else {
    *closestPosition =
        findClosestPointOnSegment(closestPosition->segment_index);
  }
  currentTrackedTarget_->segment_index = closestPosition->segment_index;
  currentTrackedTarget_->position_in_segment = closestPosition->segment_length;

  currentTrackedTarget_->movement = closestPosition->state;
  currentTrackedTarget_->lookahead = lookahead_distance;
  currentTrackedTarget_->heading_error = Angle::normalizeTo0Pi(
      currentTrackedTarget_->movement.yaw) - Angle::normalizeTo0Pi(currentState.yaw);

  currentTrackedTarget_->crosstrack_error = closestPosition->parallel_distance;
  currentTrackedTarget_->reverse =  false;
  return;
}

bool Follower::isForwardSegment(const Path::Path &segment1,
                                const Path::Path &segment2) const {

  const double angle_between_points =
      std::atan2(segment2.getStart().y - segment1.getStart().y,
                 segment2.getStart().x - segment1.getStart().x);

  return std::abs(Angle::normalizeTo0Pi(segment2.getStartOrientation() -
                                        angle_between_points)) <=
         (M_PI - std::abs(Angle::normalizeTo0Pi(
                     angle_between_points - segment1.getStartOrientation())));
}

// Get the control commands
double Follower::getLinearVelocityCmdX() const {
  return latest_velocity_command_.vx;
}
double Follower::getLinearVelocityCmdY() const {
  return latest_velocity_command_.vy;
}
double Follower::getAngularVelocityCmd() const {
  return latest_velocity_command_.omega;
}
double Follower::getSteeringAngleCmd() const {
  return latest_velocity_command_.steer_ang;
}
const double Follower::getPathLength() const {
  return currentPath->totalPathLength();
}
const bool Follower::hasPath() const {
  return currentPath->totalPathLength() > 0.0;
}
} // namespace Control
} // namespace Kompass
