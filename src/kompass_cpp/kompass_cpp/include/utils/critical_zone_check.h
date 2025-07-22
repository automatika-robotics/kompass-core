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
                      const float slowdown_distance,
                      const std::vector<double> &angles, const float min_height,
                      const float max_height, const float range_max);

  /**
   * @brief Destroy the CriticalZoneChecker object
   *
   */
  ~CriticalZoneChecker() = default;

  void preset(const std::vector<double> &angles);

  /**
   * @brief Uses laserscan data to check if the robot is in the slowdown or
   * critical zone
   *
   * @param ranges    LaserScan ranges
   * @param forward   True if the robot is moving forward, false otherwise
   * @return    Slowdown factor (0.0 - 1.0) if in the slowdown zone, 0.0 if in
   * the critical zone (stop), 1.0 otherwise
   */
  float check(const std::vector<double> &ranges, const bool forward);

  /**
   * Uses 3D point cloud data to check if robot is in slowdown or critical zone.
   *
   * @param data        Flattened point cloud data (int8), typically in XYZ
   * format.
   * @param point_step  Number of bytes between each point in the data array.
   * @param row_step    Number of bytes between each row in the data array.
   * @param height      Number of rows (height of the point cloud).
   * @param width       Number of columns (width of the point cloud).
   * @param x_offset    Offset (in bytes) to the x-coordinate within a point.
   * @param y_offset    Offset (in bytes) to the y-coordinate within a point.
   * @param z_offset    Offset (in bytes) to the z-coordinate within a point.
   * @return            A 2D occupancy grid as an Eigen::MatrixXi.
   */
  float check(const std::vector<int8_t> &data, int point_step, int row_step,
              int height, int width, float x_offset, float y_offset,
              float z_offset, const bool forward);

protected:
  double robotHeight_{1.0}, robotRadius_;
  float min_height_, max_height_, range_max_;
  float angle_right_forward_, angle_left_forward_, angle_right_backward_,
      angle_left_backward_;
  std::vector<float> sin_angles_;
  std::vector<float> cos_angles_;
  std::vector<size_t> indicies_forward_, indicies_backward_;
  float critical_distance_, slowdown_distance_;

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot
};
} // namespace Kompass
