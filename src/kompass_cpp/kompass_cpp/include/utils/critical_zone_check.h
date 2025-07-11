#pragma once

#include "utils/collision_check.h"
#include <Eigen/Core>

namespace Kompass {
/**
 * @brief Emergency (Critical) Zone Checker using LaserScan data
 *
 */
class CriticalZoneChecker {
public:
  /**
   * @brief Construct a new CriticalZoneChecker object
   *
   * @param robotShapeType    Type of the robot shape geometry
   * @param robotDimensions   Corresponding geometry dimensions
   * @param sensorPositionWRTbody         Position of the sensor w.r.t the robot
   * body - Considered constant
   * @param octreeRes         Resolution of the constructed OctTree
   */
  CriticalZoneChecker(const CollisionChecker::ShapeType robot_shape_type,
                      const std::vector<float> &robot_dimensions,
                      const Eigen::Vector3f &sensor_position_body,
                      const Eigen::Vector4f &sensor_rotation_body,
                      const float critical_angle, const float critical_distance,
                      const float slowdown_distance);

  /**
   * @brief Destroy the CriticalZoneChecker object
   *
   */
  ~CriticalZoneChecker() = default;

  void preset(const std::vector<double> &angles);

  /**
   * @brief Check if the robot is in the slowdown or critical zone
   *
   * @param ranges    LaserScan ranges
   * @param angles    LaserScan angles
   * @param forward   True if the robot is moving forward, false otherwise
   * @return    Slowdown factor (0.0 - 1.0) if in the slowdown zone, 0.0 if in
   * the critical zone (stop), 1.0 otherwise
   */
  float check(const std::vector<double> &ranges,
              const std::vector<double> &angles, const bool forward);

protected:
  double robotHeight_{1.0}, robotRadius_;
  float angle_right_forward_, angle_left_forward_, angle_right_backward_,
      angle_left_backward_;
  std::vector<size_t> indicies_forward_, indicies_backward_;
  bool preset_{false};
  float critical_distance_, slowdown_distance_;

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot

  void polarConvertLaserScanToBody(std::vector<double> &ranges,
                                   std::vector<double> &angles);
};
} // namespace Kompass
