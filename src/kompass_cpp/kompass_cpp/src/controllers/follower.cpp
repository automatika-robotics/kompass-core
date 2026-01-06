#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <string>

#include "controllers/follower.h"
#include "utils/angles.h"
#include "utils/logger.h"

using namespace std;

namespace Kompass {
namespace Control {

Follower::Follower() : Controller(), config() { setParams(config); }

void Follower::setParams(const FollowerParameters &config) {
  this->config = config;
  // Get parameters from config
  lookahead_distance = this->config.getParameter<double>("lookahead_distance");
  enable_reverse_driving =
      this->config.getParameter<bool>("enable_reverse_driving");
  goal_dist_tolerance =
      this->config.getParameter<double>("goal_dist_tolerance");
  goal_orientation_tolerance =
      this->config.getParameter<double>("goal_orientation_tolerance");
  loosing_goal_distance =
      this->config.getParameter<double>("loosing_goal_distance");
  path_segment_length_ =
      this->config.getParameter<double>("path_segment_length");
  max_point_interpolation_distance_ =
      this->config.getParameter<double>("max_point_interpolation_distance");
  speed_reg_curvature =
      this->config.getParameter<double>("speed_regulation_curvature");
  speed_reg_rotation =
      this->config.getParameter<double>("speed_regulation_angular");
  min_speed_regulation_factor =
      this->config.getParameter<double>("min_speed_regulation_factor");
  // Set rotate_in_place based on the robot type
  if (ctrType == Control::ControlType::ACKERMANN) {
    rotate_in_place = false;
  } else {
    rotate_in_place = true;
  }
  max_segment_size_ = getMaxSegmentSize();
}

Follower::Follower(const FollowerParameters &config) : Follower() {
  setParams(config);
}

size_t Follower::getMaxSegmentSize() const {
  return this->config.getParameter<double>("path_segment_length") /
             this->config.getParameter<double>(
                 "max_point_interpolation_distance") +
         1;
}

Follower::Target Follower::getTrackedTarget() const {
  return *currentTrackedTarget_;
}

const Path::Path Follower::getCurrentPath() const { return *currentPath; }

void Follower::clearCurrentPath() {
  // Delete old current path before setting new values
  if (currentPath) {
    currentPath = nullptr;
  }

  reached_goal_ = true;
  reached_yaw_ = true;
  path_processing_ = false;

  return;
}

void Follower::setCurrentPath(const Path::Path &path, const bool interpolate) {
  currentPath = std::make_unique<Path::Path>(path);

  if (interpolate) {
    currentPath->interpolate(max_point_interpolation_distance_,
                             interpolationType);
  }

  // Segment path
  currentPath->segment(path_segment_length_, max_segment_size_);

  // Get max number of segments in the path
  max_segment_index_ = currentPath->getNumSegments() - 1;

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
    return true;
  }

  const Path::Point goal_point = currentPath->getEnd();
  bool loosing_goal = false;

  const double current_goal_distance = std::hypot(
      currentState.x - goal_point.x(), currentState.y - goal_point.y());

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

  float left_distance = currentPath->distanceSquared(
      currentState, currentPath->getSegmentStart(left));
  float right_distance = currentPath->distanceSquared(
      currentState, currentPath->getSegmentStart(right));

  // In case only two points are available
  if (mid == right || mid == left) {
    if (left_distance <= right_distance) {
      return left;
    } else {
      return right;
    }
  }

  // Check closest segment of the closer end
  if (left_distance <= right_distance) {
    return findClosestSegmentIndex(left, mid);
  } else {
    return findClosestSegmentIndex(mid, right);
  }
}

Path::State Follower::projectPointOnSegment(const Path::Point &a,
                                            const Path::Point &b,
                                            double &segment_length) {
  double ab_x = b.x() - a.x();
  double ab_y = b.y() - a.y();

  double segment_heading = std::atan2(
      ab_y,
      ab_x); // + M_PI_2; // PI/2 is added to measure angle to x axis not y
  segment_length = std::sqrt(ab_x * ab_x + ab_y + ab_y);

  return Path::State(a.x(), a.y(), segment_heading);
}

Path::PathPosition Follower::findClosestPointOnSegment(size_t segment_index) {

  const Path::Path::View &segment_path = currentPath->getSegment(segment_index);

  double min_distance_squared = std::numeric_limits<float>::max();

  Path::State closest_point;

  double segment_position = 0.0; // in [0, 1]

  size_t start_index = currentPath->getSegmentStartIndex(segment_index);
  size_t point_index = 0;
  size_t closest_point_index = 0;

  // Get current segment start, end to calculate length and orientation
  Path::Point start = currentPath->getSegmentStart(segment_index);
  Path::Point end = currentPath->getSegmentEnd(segment_index);

  double segment_heading = std::atan2(end.y() - start.y(), end.x() - start.x());
  double distance_squared = 0.0f;

  for (auto projected_point : segment_path) {
    // find distance squared for faster comparision
    distance_squared = currentPath->distanceSquared(currentState, projected_point);

    if (distance_squared <= min_distance_squared) {
      min_distance_squared = distance_squared;
      closest_point = {projected_point.x(), projected_point.y(),
                       segment_heading};
      closest_point_index = point_index;
      segment_position =
          static_cast<double>(point_index) / (segment_path.getSize() - 1);
    }
    point_index++;
  }

  Path::PathPosition closest_position;
  closest_position.index = closest_point_index + start_index;
  closest_position.segment_index = segment_index;
  closest_position.segment_length = segment_position;
  closest_position.state = closest_point;
  closest_position.normal_distance = std::sqrt(min_distance_squared);

  // Compute parallel distance (signed lateral distance)
  double vec_x = currentState.x - closest_point.x;
  double vec_y = currentState.y - closest_point.y;

  // Calculate the direction vector based on the heading
  double cos_direction = std::cos(closest_point.yaw);
  double sin_direction = std::sin(closest_point.yaw);

  // crossProduct determines if the currentState is left or right to the closest
  // point
  double crossProduct = cos_direction * vec_y - sin_direction * vec_x;

  closest_position.parallel_distance = crossProduct > 0
                                           ? closest_position.normal_distance
                                           : -closest_position.normal_distance;

  return closest_position;
}

void Follower::determineTarget() {

  currentTrackedTarget_ = std::make_unique<Target>();
  LOG_DEBUG("Closest point global index", closestPosition->index,
            " its segment length = ", closestPosition->segment_length);
  // If closest position is never updated
  // OR If we reached end of a segment or end of the path -> Find new segment
  // then new point on segment
  if ((closestPosition->segment_length <= 0.0) ||
      (closestPosition->index >=
       currentPath->getSegmentEndIndex(current_segment_index_)) ||
      (closestPosition->segment_length >= 1.0)) {
    *closestPosition = findClosestPathPoint();
  }
  // If end of segment is not reached -> only find closest point on segment
  else {
    *closestPosition =
        findClosestPointOnSegment(closestPosition->segment_index);
  }
  currentTrackedTarget_->segment_index = current_segment_index_;
  currentTrackedTarget_->position_in_segment = closestPosition->segment_length;

  currentTrackedTarget_->movement = closestPosition->state;
  currentTrackedTarget_->lookahead = lookahead_distance;
  currentTrackedTarget_->heading_error =
      Angle::normalizeTo0Pi(currentTrackedTarget_->movement.yaw) -
      Angle::normalizeTo0Pi(currentState.yaw);

  currentTrackedTarget_->crosstrack_error = closestPosition->parallel_distance;
  currentTrackedTarget_->reverse = false;
  return;
}

bool Follower::isForwardSegment(const Path::Path &segment1,
                                const Path::Path &segment2) const {

  const float angle_between_points =
      std::atan2(segment2.getStart().y() - segment1.getStart().y(),
                 segment2.getStart().x() - segment1.getStart().x());

  return std::abs(Angle::normalizeTo0Pi(segment2.getStartOrientation() -
                                        angle_between_points)) <=
         (M_PI - std::abs(Angle::normalizeTo0Pi(
                     angle_between_points - segment1.getStartOrientation())));
}

double
Follower::calculateExponentialSpeedFactor(double current_angular_vel) const {
  if (!currentPath || !path_processing_) {
    return 1.0;
  }

  double curvature_sum = 0.0;
  double dist = 0.0;

  // Start iterating from the current tracked position
  size_t pt_idx = closestPosition->index;

  // Integrate curvature along the path up to lookahead distance
  for (; pt_idx < currentPath->getSize() - 1; ++pt_idx) {
    // Accumulate absolute curvature
    curvature_sum += std::abs(currentPath->getCurvature(pt_idx));

    // Accumulate distance
    double ds = Path::Path::distance(currentPath->getIndex(pt_idx),
                                     currentPath->getIndex(pt_idx + 1));
    dist += ds;

    if (dist >= lookahead_distance)
      break;
  }

  // Calculate exponential factor
  // factor = exp(-(speed_reg_curvature * sum(|K|) + speed_reg_rotation *
  // |omega|))
  double exponent = (speed_reg_curvature * curvature_sum) +
                    (speed_reg_rotation * std::abs(current_angular_vel));

  // restrict by the minimum allowed speed regulation factor
  return std::max(std::exp(-exponent), min_speed_regulation_factor);
}

} // namespace Control
} // namespace Kompass
