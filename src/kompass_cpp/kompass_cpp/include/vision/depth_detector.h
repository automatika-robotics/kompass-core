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

  void updateBoxes(const Eigen::MatrixX<unsigned short> aligned_depth_img,
                   const std::vector<Bbox2D> &detections, const std::optional<Path::State> &robot_state = std::nullopt);

  std::optional<std::vector<Bbox3D>> get3dDetections() const;

private:
  float cx_, cy_, fx_, fy_;         // Depth Image camera intrinsics
  float minDepth_, maxDepth_, depthConversionFactor_;
  Eigen::MatrixX<unsigned short> alignedDepthImg_;
  Eigen::Isometry3f camera_in_body_tf_, body_in_world_tf_;
  std::unique_ptr<std::vector<Bbox3D>> boxes_;

  std::optional<Bbox3D> convert2Dboxto3Dbox(const Bbox2D &box2d);

  static void calculateMAD(const std::vector<float>& depthValues, float& median, float& mad);

  static float getMedian(const std::vector<float> &values);
};

} // namespace Kompass
