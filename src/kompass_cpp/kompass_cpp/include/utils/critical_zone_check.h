#pragma once

#include "datatypes/sensors.h"
#include "datatypes/span.h"
#include "utils/collision_check.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <vector>

namespace Kompass {
/**
 * @brief Emergency (Critical) Zone Checker using LaserScan or PointCloud data
 *
 */
class CriticalZoneChecker {
public:
  // Enum to explicitly select operation mode
  enum class InputType {
    LASERSCAN, // 2D Ranges + Angles (requires pre-computation)
    POINTCLOUD // Raw 3D Bytes (XYZ)
  };

  /**
   * @brief Construct a new CriticalZoneChecker object
   *
   * @param input_type        Selects LASERSCAN or POINTCLOUD mode.
   * - LASERSCAN: requires exactly one sensor and non-empty `scan_angles`;
   * - POINTCLOUD: accepts N sensors (clouds are checked per sensor and the
   *   minimum safety factor wins); `scan_angles` is unused.
   * @param robot_shape_type  Type of the robot shape geometry
   * @param robot_dimensions  Corresponding geometry dimensions
   * @param sensors           One SensorConfig per sensor (mount pose in the
   * body frame + the point field encoding for pointcloud input)
   * @param critical_angle    Full angle of the safety cone (degrees)
   * @param critical_distance Distance for emergency stop (m)
   * @param slowdown_distance Distance for linear slowdown (m)
   * @param min_height        Minimum accepted point height (m). For
   * pointcloud input this is a BODY-frame band shared by all sensors
   * (typically 0 .. robot_height): each sensor's mount transform is applied
   * to the point before the gate
   * @param max_height        Maximum accepted point height (m), body-frame
   * for pointcloud input (see min_height)
   * @param range_max         Maximum valid sensor range (m)
   * @param scan_angles       (LASERSCAN only) scan angles in radians
   */
  CriticalZoneChecker(InputType input_type,
                      const CollisionChecker::ShapeType robot_shape_type,
                      const std::vector<float> &robot_dimensions,
                      const std::vector<SensorConfig> &sensors,
                      const float critical_angle, const float critical_distance,
                      const float slowdown_distance, const float min_height,
                      const float max_height, const float range_max,
                      const std::vector<double> &scan_angles = {});

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
   *
   * @throws std::logic_error when the checker was constructed for pointcloud
   * input.
   */
  float check(Eigen::Ref<const Eigen::VectorXf> ranges, const bool forward);

  /**
   * @brief Checks N point clouds (one per configured sensor, positional
   * pairing) and returns the minimum safety factor across all of them.
   *
   * Each surviving point is transformed with its sensor's full mount
   * isometry, gated on BODY-frame height, filtered by the body-frame
   * critical cone (forward or backward), and scored by its planar distance
   * to the robot's bounding circle. Empty views are skipped; an all-empty batch
   * returns 1.0 (no constraint).
   *
   * @param clouds    One view per configured sensor
   * @param forward   True if the robot is moving forward, false otherwise
   * @return    Slowdown factor (0.0 - 1.0); 0.0 = emergency stop
   *
   * @throws std::logic_error when the checker was not constructed for
   * pointcloud input.
   * @throws std::invalid_argument on cloud count mismatch or negative field
   * offsets (message names the offending cloud index).
   */
  float check(Span<PointCloudView> clouds, const bool forward);

  // Convenience overload for containers / braced lists
  float check(const std::vector<PointCloudView> &clouds, const bool forward) {
    return check(Span<PointCloudView>(clouds), forward);
  }

  /**
   * Single-cloud adapter onto the batched check above (the checker must be
   * configured with exactly one sensor).
   *
   * @param data        Flattened point cloud data (uint8), typically in XYZ
   * format.
   * @param point_step  Number of bytes between each point in the data array.
   * @param row_step    Number of bytes between each row in the data array.
   * @param height      Number of rows (height of the point cloud).
   * @param width       Number of columns (width of the point cloud).
   * @param x_offset    Offset (in bytes) to the x-coordinate within a point.
   * @param y_offset    Offset (in bytes) to the y-coordinate within a point.
   * @param z_offset    Offset (in bytes) to the z-coordinate within a point.
   * @param forward     True if the robot is moving forward
   * @return            Slowdown factor (0.0 - 1.0); 0.0 = emergency stop
   */
  float check(ByteSpan data, int point_step, int row_step, int height,
              int width, int x_offset, int y_offset, int z_offset,
              const bool forward);

  /// Number of configured sensors
  size_t numSensors() const { return sensors_.size(); }

protected:
  // Per-sensor runtime state derived once from a SensorConfig at
  // construction; nothing here changes per tick
  struct SensorRuntime {
    Eigen::Isometry3f tf_body; // sensor -> body mount (laserscan per-beam)
    // Rows 0..2 of tf_body [R | t], row-major
    std::array<float, 12> tf;
    PointFieldType field_type; // point field encoding (pointcloud decode)
    int elem_size;             // sizeof one field element
  };

  InputType input_type_;
  double robotHeight_{1.0}, robotRadius_;
  float min_height_, max_height_, range_max_;
  float critical_angle_;
  std::vector<float> sin_angles_;
  std::vector<float> cos_angles_;
  std::vector<size_t> indicies_forward_, indicies_backward_;
  float critical_distance_, slowdown_distance_;

  // One runtime per configured sensor (laserscan input has exactly one)
  std::vector<SensorRuntime> sensors_;

  // Init time derived constants

  // squared radius of the outermost ring that can matter (squared so the
  // per-point rejection needs no sqrt)
  float slow_limit_sq_;
  // slowdown ramp's inverse width (turns the per-point division into a
  // multiply)
  float inv_dist_range_;
};
} // namespace Kompass
