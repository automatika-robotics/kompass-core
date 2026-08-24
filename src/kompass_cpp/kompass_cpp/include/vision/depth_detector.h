/**
 * @brief Implementation based on the following work [Xu2024OnboardDO]
 *
 * "Onboard Dynamic-Object Detection and Tracking for Autonomous Robot
 * Navigation With RGB-D Camera" Z. Xu, X. Zhan, Y. Xiu, C. Suzuki, and K.
 * Shimada. IEEE Robotics and Automation Letters, vol. 9, no. 1, pp. 651–658,
 * 2024. doi:10.1109/LRA.2023.3334683
 *
 * @article{Xu2024OnboardDO,
 *   title   = {Onboard Dynamic-Object Detection and Tracking for Autonomous
 * Robot Navigation With RGB-D Camera}, author  = {Z. Xu and X. Zhan and Y. Xiu
 * and C. Suzuki and K. Shimada}, journal = {IEEE Robotics and Automation
 * Letters}, volume  = {9}, number  = {1}, pages   = {651--658}, year    =
 * {2024}, doi     = {10.1109/LRA.2023.3334683}, keywords = { Detectors,
 * Cameras, Three-dimensional displays, Point cloud compression, Robot vision
 * systems, Heuristic algorithms, Collision avoidance, RGB-D perception,
 * Vision-based navigation, Visual tracking, 3D object detection
 *   }
 * }
 */

#pragma once

#include "datatypes/path.h"
#include "datatypes/sensors.h"
#include "datatypes/tracking.h"
#include <Eigen/Dense>
#include <Eigen/src/Geometry/Transform.h>
#include <optional>
#include <vector>

namespace Kompass {

class DepthDetector {
public:
  /**
   * @brief Axis convention the camera pose handed to the constructor is
   * expressed in.
   *
   * ROS names a camera's optical frame in the header of every Image and
   * CameraInfo message, and that is the frame TF can resolve, so `Optical` is
   * the default: a pose looked up straight out of the ROS graph is correct
   * with no further handling. `BodyAligned` is for callers that have already
   * turned the pose into robot axes themselves.
   *
   * The two differ by the fixed REP 103 quarter rotation, so passing the wrong
   * one puts every detection 90 degrees off.
   */
  enum class CameraFrameConvention {
    Optical,    ///< x right, y down, z into the image (REP 103 optical)
    BodyAligned ///< x forward, y left, z up (REP 103 body)
  };

  DepthDetector(const Eigen::Vector2f &depth_range,
                const Eigen::Vector3f &camera_in_body_translation,
                const Eigen::Quaternionf &camera_in_body_rotation,
                const Eigen::Vector2f &focal_length,
                const Eigen::Vector2f &principal_point,
                const float depth_conversion_factor = 1e-3,
                const CameraFrameConvention convention =
                    CameraFrameConvention::Optical);

  DepthDetector(const Eigen::Vector2f &depth_range,
                const Eigen::Isometry3f &camera_in_body_tf,
                const Eigen::Vector2f &focal_length,
                const Eigen::Vector2f &principal_point,
                const float depth_conversion_factor = 1e-3,
                const CameraFrameConvention convention =
                    CameraFrameConvention::Optical);

  /// REP 103 rotation taking optical axes to body-aligned ones.
  static const Eigen::Quaternionf &opticalToBodyAligned();

  /**
   * NOTE: Non owning zero-copy view of the depth image. UINT16 pixels are
   * scaled by the configured depth_conversion_factor (mm -> m by default);
   * FLOAT32 pixels are taken as metres. Non-finite pixels are rejected by the
   * min/max range gate.
   */
  void
  updateBoxes(const DepthImageView &aligned_depth_img,
              const std::vector<Bbox2D> &detections,
              const std::optional<Path::State> &robot_state = std::nullopt);

  void updatePOIs(const DepthImageView &aligned_depth_img,
                  const PointsOfInterest &pois,
                  const std::optional<Path::State> &robot_state = std::nullopt);

  /// Result of the last update; empty when no box converted.
  const std::vector<Bbox3D> &get3dDetections() const { return boxes_; }

private:
  float cx_, cy_, fx_, fy_; // Depth Image camera intrinsics
  float minDepth_, maxDepth_, depthConversionFactor_;
  Eigen::Isometry3f camera_in_body_tf_, body_in_world_tf_;
  std::vector<Bbox3D> boxes_;

  // Per-box scratch, refilled for every converted box. Keep the
  // allocated capacity, so it grows to the largest box seen.
  std::vector<float> depth_values_; // in-range depth values (metres)
  std::vector<float> mad_scratch_; // absolute deviations from the median

  std::optional<Bbox3D> convert2Dboxto3Dbox(const DepthImageView &depth,
                                            const Bbox2D &box2d);

  std::optional<Bbox3D> convertPOIto3Dbox(const DepthImageView &depth,
                                          const PointsOfInterest &poi);

  void calculateMAD(std::vector<float> &depthValues, float &median, float &mad);

  /// Median via nth_element selection (O(n)).
  /// for even n the result averages the two middle order statistics
  static float getMedian(std::vector<float> &values);
};

} // namespace Kompass
