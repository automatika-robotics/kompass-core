#include "utils/critical_zone_check.h"
#include "utils/angles.h"
#include "utils/logger.h"
#include "utils/pointcloud.h"
#include "utils/transformation.h"
#include <Eigen/Core>

namespace Kompass {
/**
 * @brief Emergency (Critical) Zone Checker using LaserScan data
 *
 */

CriticalZoneChecker::CriticalZoneChecker(
    const CollisionChecker::ShapeType robot_shape_type,
    const std::vector<float> &robot_dimensions,
    const Eigen::Vector3f &sensor_position_body,
    const Eigen::Vector4f &sensor_rotation_body, const float critical_angle,
    const float critical_distance, const float slowdown_distance,
    const std::vector<double> &angles, const float min_height,
    const float max_height, const float range_max) {
  min_height_ = min_height;
  max_height_ = max_height;
  range_max_ = range_max;
  // Construct  a geometry object based on the robot shape
  if (robot_shape_type == CollisionChecker::ShapeType::CYLINDER) {
    robotHeight_ = robot_dimensions.at(1);
    robotRadius_ = robot_dimensions.at(0);
  } else if (robot_shape_type == CollisionChecker::ShapeType::BOX) {
    robotHeight_ = robot_dimensions.at(2);
    robotRadius_ = std::sqrt(pow(robot_dimensions.at(0), 2) +
                             pow(robot_dimensions.at(1), 2)) /
                   2;
  } else {
    throw std::invalid_argument("Invalid robot geometry type");
  }

  // Init the sensor position w.r.t body
  sensor_tf_body_ =
      getTransformation(sensor_rotation_body, sensor_position_body);
  // Compute the critical zone angles min,max
  float angle_rad = critical_angle * M_PI / 180.0;
  angle_right_forward_ = angle_rad / 2;
  angle_left_forward_ = (2 * M_PI) - (angle_rad / 2);
  angle_right_backward_ = Angle::normalizeTo0Pi(M_PI + angle_right_forward_);
  angle_left_backward_ = Angle::normalizeTo0Pi(M_PI + angle_left_forward_);

  LOG_DEBUG("Critical zone forward angles: [", angle_right_forward_, ", ",
            angle_left_forward_, "]");
  LOG_DEBUG("Critical zone backward angles: [", angle_right_backward_, ", ",
            angle_left_backward_, "]");

  preset(angles);

  // Set critical distance
  if (slowdown_distance <= critical_distance) {

    throw std::invalid_argument(
        "SlowDown distance must be greater than the Critical distance!");
  }
  critical_distance_ = critical_distance;
  slowdown_distance_ = slowdown_distance;
}

void CriticalZoneChecker::preset(const std::vector<double> &angles) {
  Eigen::Vector3f cartesianPoint;
  float theta;
  sin_angles_.resize(angles.size());
  cos_angles_.resize(angles.size());

  for (size_t i = 0; i < angles.size(); ++i) {
    cos_angles_[i] = std::cos(angles[i]);
    sin_angles_[i] = std::sin(angles[i]);
    cartesianPoint = {cos_angles_[i], sin_angles_[i], 0.0f};
    // Apply TF
    cartesianPoint = sensor_tf_body_ * cartesianPoint;

    // check if within the zone
    theta = Angle::normalizeTo0Pi(
        std::atan2(cartesianPoint.y(), cartesianPoint.x()));

    if (theta >= std::max(angle_left_forward_, angle_right_forward_) ||
        theta <= std::min(angle_left_forward_, angle_right_forward_)) {
      indicies_forward_.push_back(i);
    }
    if (theta >= std::min(angle_left_backward_, angle_right_backward_) &&
        theta <= std::max(angle_left_backward_, angle_right_backward_)) {
      indicies_backward_.push_back(i);
    }
  }
}

float CriticalZoneChecker::check(const std::vector<double> &ranges,
                                 const bool forward) {
  std::vector<size_t> *indicies;
  float x, y, converted_range;
  Eigen::Vector3f cartesianPoint;
  if (forward) {
    indicies = &indicies_forward_;
  } else {
    indicies = &indicies_backward_;
  }
  // If sensor data has been preset then use the indicies directly
  for (size_t index : *indicies) {
    x = ranges[index] * cos_angles_[index];
    y = ranges[index] * sin_angles_[index];
    cartesianPoint = {x, y, 0.0f};
    // Apply TF
    cartesianPoint = sensor_tf_body_ * cartesianPoint;

    // check if within the zone
    converted_range = std::sqrt(std::pow(cartesianPoint.y(), 2) +
                                std::pow(cartesianPoint.x(), 2));
    float distance = converted_range - robotRadius_;
    if (distance <= critical_distance_) {
      return 0.0;
    } else if (distance <= slowdown_distance_) {
      return (distance - critical_distance_) /
             (slowdown_distance_ - critical_distance_);
    }
  }
  return 1.0;
}

float CriticalZoneChecker::check(const std::vector<int8_t> &data,
                                 int point_step, int row_step, int height,
                                 int width, float x_offset, float y_offset,
                                 float z_offset, const bool forward) {

  std::vector<double> ranges;
  pointCloudToLaserScanFromRaw(
      data, point_step, row_step, height, width, x_offset, y_offset, z_offset,
      range_max_, min_height_, max_height_, sin_angles_.size(), ranges);
  return check(ranges, forward);
}
} // namespace Kompass
