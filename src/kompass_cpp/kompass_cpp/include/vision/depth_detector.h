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
                const Eigen::Vector3f &body_to_camera_translation,
                const Eigen::Quaternionf &body_to_camera_rotation,
                const Eigen::Vector2f &focal_length,
                const Eigen::Vector2f &principal_point,
                const float depth_conversion_factor = 1e-3);

  DepthDetector(const Eigen::Vector2f &depth_range,
                const Eigen::Isometry3f &body_to_camera_tf,
                const Eigen::Vector2f &focal_length,
                const Eigen::Vector2f &principal_point,
                const float depth_conversion_factor = 1e-3);

  void updateState(const Path::State& current_state);

  void updateState(const Eigen::Isometry3f &robot_tf);

  void updateBoxes(const Eigen::MatrixXi aligned_depth_img, const std::vector<Bbox2D>& detections);

  std::optional<std::vector<Bbox3D>> get3dDetections() const;

private:
  float cx_, cy_, fx_, fy_;         // Depth Image camera intrinsics
  float minDepth_, maxDepth_, depthConversionFactor_;
  Eigen::MatrixXi alignedDepthImg_;
  Eigen::Isometry3f body_to_camera_tf_, world_to_body_tf_;
  std::unique_ptr<std::vector<Bbox3D>> boxes_;

  std::optional<Bbox3D> convert2Dboxto3Dbox(const Bbox2D &box2d);

  static void calculateMAD(const std::vector<float>& depthValues, float& median, float& mad);

  static float getMedian(const std::vector<float> &values);
};

} // namespace Kompass
