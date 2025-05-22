#include "utils/critical_zone_check.h"
#include "utils/angles.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <Eigen/Core>
#include <array>

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
    const float critical_distance, const float slowdown_distance) {
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
  float x, y, theta;

  for (size_t i = 0; i < angles.size(); ++i) {
    x = std::cos(angles[i]);
    y = std::sin(angles[i]);
    cartesianPoint = {x, y, 0.0f};
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
  preset_ = true;
}

float CriticalZoneChecker::check(const std::vector<double> &ranges,
                                 const std::vector<double> &angles,
                                 const bool forward) {
  if (angles.size() != ranges.size()) {
    LOG_ERROR("Angles and ranges vectors must have the same size!");
    return false;
  }
  if (!preset_) {
    preset(angles);
    return check(ranges, angles, forward);
  } else {
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
      if (index < ranges.size()) {
        x = ranges[index] * std::cos(angles[index]);
        y = ranges[index] * std::sin(angles[index]);
        cartesianPoint = {x, y, 0.0f};
        // Apply TF
        cartesianPoint = sensor_tf_body_ * cartesianPoint;
        auto transformation_matrix = sensor_tf_body_.matrix();
        const std::array<float, 4> x_transform{
            transformation_matrix(0, 0), transformation_matrix(0, 1),
            transformation_matrix(0, 2), transformation_matrix(0, 3)};

        const std::array<float, 4> y_transform{
            transformation_matrix(1, 0), transformation_matrix(1, 1),
            transformation_matrix(1, 2), transformation_matrix(1, 3)};

        std::array<float, 4> point{x,
                                   y, 0.0, 1.0};
        std::array<float, 2> transformed_point{0.0, 0.0};

        for (size_t i = 0; i < 4; ++i) {
          transformed_point[0] += x_transform[i] * point[i];
          transformed_point[1] += y_transform[i] * point[i];
        }

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
      } else {
        preset_ = false;
        return check(ranges, angles, forward);
      }
    }
    return 1.0;
  }
}

void CriticalZoneChecker::polarConvertLaserScanToBody(
    std::vector<double> &ranges, std::vector<double> &angles) {
  if (angles.size() != ranges.size()) {
    LOG_ERROR("Angles and ranges vectors must have the same size!");
    return;
  }

  Eigen::Vector3f cartesianPoint;
  float x, y;

  for (size_t i = 0; i < angles.size(); i++) {
    x = ranges[i] * std::cos(angles[i]);
    y = ranges[i] * std::sin(angles[i]);
    cartesianPoint = {x, y, 0.0f};
    // Apply TF
    cartesianPoint = sensor_tf_body_ * cartesianPoint;

    // convert back to polar
    angles[i] = Angle::normalizeTo0Pi(
        std::atan2(cartesianPoint.y(), cartesianPoint.x()));
    ranges[i] = cartesianPoint.norm();
  }
}

} // namespace Kompass
