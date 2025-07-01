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
#include "datatypes/tracking.h"
#include <Eigen/Dense>
#include <Eigen/src/Geometry/Transform.h>
#include <memory>
#include <optional>
#include <vector>

namespace Kompass {

class DepthDetector {
public:
  DepthDetector(const Eigen::Vector2f &depth_range,
                const Eigen::Vector3f &camera_in_body_translation,
                const Eigen::Quaternionf &camera_in_body_rotation,
                const Eigen::Vector2f &focal_length,
                const Eigen::Vector2f &principal_point,
                const float depth_conversion_factor = 1e-3);

  DepthDetector(const Eigen::Vector2f &depth_range,
                const Eigen::Isometry3f &camera_in_body_tf,
                const Eigen::Vector2f &focal_length,
                const Eigen::Vector2f &principal_point,
                const float depth_conversion_factor = 1e-3);

  void
  updateBoxes(const Eigen::MatrixX<unsigned short> aligned_depth_img,
              const std::vector<Bbox2D> &detections,
              const std::optional<Path::State> &robot_state = std::nullopt);

  std::optional<std::vector<Bbox3D>> get3dDetections() const;

private:
  float cx_, cy_, fx_, fy_; // Depth Image camera intrinsics
  float minDepth_, maxDepth_, depthConversionFactor_;
  Eigen::MatrixX<unsigned short> alignedDepthImg_;
  Eigen::Isometry3f camera_in_body_tf_, body_in_world_tf_;
  std::unique_ptr<std::vector<Bbox3D>> boxes_;

  std::optional<Bbox3D> convert2Dboxto3Dbox(const Bbox2D &box2d);

  static void calculateMAD(const std::vector<float> &depthValues, float &median,
                           float &mad);

  static float getMedian(const std::vector<float> &values);
};

} // namespace Kompass
