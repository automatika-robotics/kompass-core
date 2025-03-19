#include "utils/critical_zone_check.h"
#include "utils/angles.h"
#include "utils/logger.h"
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
    const std::array<float, 3> &sensor_position_body,
    const std::array<float, 4> &sensor_rotation_body,
    const float critical_angle, const float critical_distance) {
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
      getTransformation(Eigen::Quaternionf(sensor_rotation_body.data()),
                        Eigen::Vector3f(sensor_position_body.data()));
  // Compute the critical zone angles min,max
  float angle_rad = critical_angle * M_PI / 180.0;
  angle_right_forward_ = angle_rad / 2;
  angle_left_forward_ = (2 * M_PI) - (angle_rad / 2);
  angle_right_backward_ = Angle::normalizeTo0Pi(M_PI + angle_right_forward_);
  angle_left_backward_ = Angle::normalizeTo0Pi(M_PI + angle_left_forward_);

  LOG_INFO("angles forward  ", angle_right_forward_, ", ", angle_left_forward_);
  LOG_INFO("angles backward  ", angle_right_backward_, ", ",
           angle_left_backward_);

  // Set critical distance
  critical_distance_ = critical_distance;
}

void CriticalZoneChecker::preset(const std::vector<double> &angles) {
  Eigen::Vector3f cartesianPoint;
  float x, y, theta, converted_range;

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

bool CriticalZoneChecker::check(const std::vector<double> &ranges,
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

        // check if within the zone
        converted_range = std::sqrt(std::pow(cartesianPoint.y(), 2) +
                                    std::pow(cartesianPoint.x(), 2));

        if (converted_range - robotRadius_ <= critical_distance_) {
          return true;
        }
      } else {
        preset_ = false;
        return check(ranges, angles, forward);
      }
    }
    return false;
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
