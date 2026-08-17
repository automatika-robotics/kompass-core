#include "utils/critical_zone_check.h"
#include "utils/angles.h"
#include <Eigen/Core>
#include <stdexcept>

namespace Kompass {
/**
 * @brief Emergency (Critical) Zone Checker using LaserScan or PointCloud data
 *
 */

CriticalZoneChecker::CriticalZoneChecker(
    InputType input_type, const CollisionChecker::ShapeType robot_shape_type,
    const std::vector<float> &robot_dimensions,
    const std::vector<SensorConfig> &sensors, const float critical_angle,
    const float critical_distance, const float slowdown_distance,
    const float min_height, const float max_height, const float range_max,
    const std::vector<double> &scan_angles) {
  input_type_ = input_type;
  min_height_ = min_height;
  max_height_ = max_height;
  range_max_ = range_max;
  // Size the robot through the shared derivation so the critical zone and the
  // collision checker cannot disagree about the same robot
  robotRadius_ = CollisionChecker::radiusOf(robot_shape_type, robot_dimensions);
  robotHeight_ = CollisionChecker::heightOf(robot_shape_type, robot_dimensions);

  if (sensors.empty()) {
    throw std::invalid_argument(
        "CriticalZoneChecker requires at least one sensor config");
  }
  if (input_type_ == InputType::LASERSCAN) {
    // No fusion in lasercan path
    if (sensors.size() != 1) {
      throw std::invalid_argument(
          "CriticalZoneChecker laserscan input supports exactly one sensor");
    }
    if (scan_angles.empty()) {
      throw std::invalid_argument(
          "CriticalZoneChecker laserscan input requires non-empty "
          "scan_angles");
    }
  }
  sensors_.reserve(sensors.size());
  for (const auto &sensor : sensors) {
    SensorRuntime runtime;
    runtime.tf_body = sensor.tfBody();
    const Eigen::Matrix4f tf = runtime.tf_body.matrix();
    runtime.tf = {tf(0, 0), tf(0, 1), tf(0, 2), tf(0, 3),
                  tf(1, 0), tf(1, 1), tf(1, 2), tf(1, 3),
                  tf(2, 0), tf(2, 1), tf(2, 2), tf(2, 3)};
    runtime.field_type = sensor.cloud_field_type;
    runtime.elem_size = elementSizeOf(sensor.cloud_field_type);
    sensors_.push_back(runtime);
  }

  // Compute the normalized critical zone angle
  float angle_rad = critical_angle * M_PI / 180.0;
  critical_angle_ = Angle::normalizeToMinusPiPlusPi(angle_rad / 2);

  if (input_type_ == InputType::LASERSCAN) {
    preset(scan_angles);
  }

  // Set critical distance
  if (slowdown_distance <= critical_distance) {

    throw std::invalid_argument(
        "SlowDown distance must be greater than the Critical distance!");
  }
  critical_distance_ = critical_distance;
  slowdown_distance_ = slowdown_distance;

  // Derived thresholds
  const float slow_limit =
      slowdown_distance_ + static_cast<float>(robotRadius_);
  slow_limit_sq_ = slow_limit * slow_limit;
  inv_dist_range_ = 1.0f / (slowdown_distance_ - critical_distance_);
}

void CriticalZoneChecker::preset(const std::vector<double> &angles) {
  Eigen::Vector3f cartesianPoint;
  float abs_theta;
  sin_angles_.resize(angles.size());
  cos_angles_.resize(angles.size());
  // Recompute from scratch. Stale sets from a previous call must not survive
  indicies_forward_.clear();
  indicies_backward_.clear();

  for (size_t i = 0; i < angles.size(); ++i) {
    cos_angles_[i] = std::cos(angles[i]);
    sin_angles_[i] = std::sin(angles[i]);
    cartesianPoint = {cos_angles_[i], sin_angles_[i], 0.0f};
    // Apply TF
    cartesianPoint = sensors_[0].tf_body * cartesianPoint;

    // check if within the zone
    abs_theta = std::abs(std::atan2(cartesianPoint.y(), cartesianPoint.x()));

    if (abs_theta <= critical_angle_) {
      indicies_forward_.push_back(i);
    }
    if (abs_theta >= M_PI - critical_angle_) {
      indicies_backward_.push_back(i);
    }
  }
}

float CriticalZoneChecker::check(Eigen::Ref<const Eigen::VectorXf> ranges,
                                 const bool forward) {
  if (input_type_ != InputType::LASERSCAN) {
    throw std::logic_error(
        "check(ranges): checker was constructed for pointcloud input");
  }
  std::vector<size_t> *indicies;
  float x, y, converted_range;
  Eigen::Vector3f cartesianPoint;
  if (forward) {
    indicies = &indicies_forward_;
  } else {
    indicies = &indicies_backward_;
  }
  // If sensor data has been preset then use the indicies directly
  float slowdown_factor = 1.0f;
  for (size_t index : *indicies) {
    x = ranges[index] * cos_angles_[index];
    y = ranges[index] * sin_angles_[index];
    cartesianPoint = {x, y, 0.0f};
    // Apply TF
    cartesianPoint = sensors_[0].tf_body * cartesianPoint;

    // Coarse squared rejection. Beams beyond the slowdown ring get skipped
    const float dist_sq = cartesianPoint.x() * cartesianPoint.x() +
                          cartesianPoint.y() * cartesianPoint.y();
    if (dist_sq > slow_limit_sq_) {
      continue;
    }
    converted_range = std::sqrt(dist_sq);
    float distance = converted_range - robotRadius_;
    if (distance <= critical_distance_) {
      return 0.0;
    } else if (distance <= slowdown_distance_) {
      slowdown_factor = std::min(
          slowdown_factor, (distance - critical_distance_) * inv_dist_range_);
    }
  }
  return slowdown_factor;
}

float CriticalZoneChecker::check(Span<PointCloudView> clouds,
                                 const bool forward) {
  if (input_type_ != InputType::POINTCLOUD) {
    throw std::logic_error(
        "check(clouds): checker was not constructed for pointcloud input");
  }
  validateClouds(clouds, sensors_.size());

  float min_factor = 1.0f;
  for (size_t s = 0; s < clouds.size(); ++s) {
    const auto &cloud = clouds[s];
    if (cloud.empty()) {
      continue; // no data from this sensor this tick
    }
    // Per-point body-frame walk
    const auto &sensor = sensors_[s];
    // Get tf_body values
    const float t00 = sensor.tf[0], t01 = sensor.tf[1], t02 = sensor.tf[2],
                t03 = sensor.tf[3];
    const float t10 = sensor.tf[4], t11 = sensor.tf[5], t12 = sensor.tf[6],
                t13 = sensor.tf[7];
    const float t20 = sensor.tf[8], t21 = sensor.tf[9], t22 = sensor.tf[10],
                t23 = sensor.tf[11];
    // Get scan params
    const PointFieldType field_type = sensor.field_type;
    const int elem_size = sensor.elem_size;
    const int max_offset =
        std::max({cloud.x_offset, cloud.y_offset, cloud.z_offset});
    const int row_bytes = cloud.width * cloud.point_step;

    for (int row = 0; row < cloud.height; ++row) {
      for (int col = 0; col < row_bytes; col += cloud.point_step) {
        const std::size_t point_start = row * cloud.row_step + col;
        if (point_start + max_offset + elem_size > cloud.data.size()) {
          continue;
        }

        // Grab x, y, z
        const float x = load_and_cast_val(
            cloud.data.data(), point_start + cloud.x_offset, field_type);
        const float y = load_and_cast_val(
            cloud.data.data(), point_start + cloud.y_offset, field_type);
        const float z = load_and_cast_val(
            cloud.data.data(), point_start + cloud.z_offset, field_type);

        // Reject non-finite points (NaN padding in organized clouds)
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }

        // Filter sensor points with no planar extent in the SENSOR frame. Maps
        // onto the mount position (inside the robot) and must never trigger a
        // stop
        if (x * x + y * y < 1e-6f) {
          continue;
        }

        // Full mount transform into the body frame
        const float xb = t00 * x + t01 * y + t02 * z + t03;
        const float yb = t10 * x + t11 * y + t12 * z + t13;
        const float zb = t20 * x + t21 * y + t22 * z + t23;

        // Body-frame height band
        if (zb < min_height_ || zb > max_height_) {
          continue;
        }

        // Coarse distance rejection for points beyond slow down distance
        const float dist_sq = xb * xb + yb * yb;
        if (dist_sq > slow_limit_sq_) {
          continue;
        }

        // Body-frame critical cone (forward: |angle| <= half-cone;
        // backward: |angle| >= pi - half-cone)
        const float abs_angle = std::fabs(std::atan2(yb, xb));
        const bool in_zone = forward ? (abs_angle <= critical_angle_)
                                     : (abs_angle >= M_PI - critical_angle_);
        if (!in_zone) {
          continue;
        }

        const float dist_to_robot =
            std::sqrt(dist_sq) - static_cast<float>(robotRadius_);
        if (dist_to_robot <= critical_distance_) {
          return 0.0f; // emergency stop, early exit
        }
        if (dist_to_robot <= slowdown_distance_) {
          // slowdown factor
          min_factor =
              std::min(min_factor,
                       (dist_to_robot - critical_distance_) * inv_dist_range_);
        }
      }
    }
  }
  return min_factor;
}

// Single pointcloud overload
float CriticalZoneChecker::check(ByteSpan data, int point_step, int row_step,
                                 int height, int width, int x_offset,
                                 int y_offset, int z_offset,
                                 const bool forward) {
  // Allocation-free N=1 adapter
  const PointCloudView view{data,  point_step, row_step, height,
                            width, x_offset,   y_offset, z_offset};
  return check(Span<PointCloudView>(&view, 1), forward);
}
} // namespace Kompass
