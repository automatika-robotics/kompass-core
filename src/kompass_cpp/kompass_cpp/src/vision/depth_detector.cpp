#include "vision/depth_detector.h"
#include "datatypes/tracking.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <memory>
#include <optional>
#include <vector>

namespace Kompass {

DepthDetector::DepthDetector(const Eigen::Vector2f &depth_range,
                             const Eigen::Vector3f &camera_in_body_translation,
                             const Eigen::Quaternionf &camera_in_body_rotation,
                             const Eigen::Vector2f &focal_length,
                             const Eigen::Vector2f &principal_point,
                             const float depth_conversion_factor)
    : DepthDetector(depth_range,
                    getTransformation(camera_in_body_rotation,
                                      camera_in_body_translation),
                    focal_length, principal_point, depth_conversion_factor) {}

DepthDetector::DepthDetector(
    const Eigen::Vector2f &depth_range,
    const Eigen::Isometry3f &camera_in_body_tf,
    const Eigen::Vector2f &focal_length, const Eigen::Vector2f &principal_point,
    const float depth_conversion_factor) { // Range of interest for depth values
                                           // in meters
  minDepth_ = depth_range(0);
  maxDepth_ = depth_range(1);
  // Factor to convert depth image data to meters (in ROS2 its given in mm ->
  // depthConversionFactor = 1e-3)
  depthConversionFactor_ = depth_conversion_factor;
  // Set camera tf
  camera_in_body_tf_ = camera_in_body_tf;

  // Set camera  intrinsic parameters
  fx_ = focal_length.x();
  fy_ = focal_length.y();
  cx_ = principal_point.x();
  cy_ = principal_point.y();

  body_in_world_tf_ = Eigen::Isometry3f::Identity();
}

std::optional<std::vector<Bbox3D>> DepthDetector::get3dDetections() const {
  if (boxes_) {
    return *boxes_;
  }
  return std::nullopt;
}

void DepthDetector::updateBoxes(
    const Eigen::MatrixX<unsigned short> &aligned_depth_img,
    const std::vector<Bbox2D> &detections,
    const std::optional<Path::State> &robot_state) {
  if (robot_state.has_value()) {
    body_in_world_tf_ = getTransformation(robot_state.value());
  }
  alignedDepthImg_ = aligned_depth_img;
  boxes_ = std::make_unique<std::vector<Bbox3D>>();
  for (const auto &box2d : detections) {
    auto converted_box = convert2Dboxto3Dbox(box2d);
    if (converted_box) {
      boxes_->push_back(converted_box.value());
    }
  }
}

void DepthDetector::updatePOIs(
    const Eigen::MatrixX<unsigned short> &aligned_depth_img,
    const PointsOfInterest &poi,
    const std::optional<Path::State> &robot_state) {
  if (robot_state.has_value()) {
    body_in_world_tf_ = getTransformation(robot_state.value());
  }
  alignedDepthImg_ = aligned_depth_img;
  boxes_ = std::make_unique<std::vector<Bbox3D>>();
  auto converted_box = convertPOIto3Dbox(poi);
  if (converted_box) {
    boxes_->push_back(converted_box.value());
  }
}

std::optional<Bbox3D> DepthDetector::convert2Dboxto3Dbox(const Bbox2D &box2d) {
  Bbox3D box3d(box2d);
  Eigen::Vector2i x_limits = box2d.getXLimits();
  Eigen::Vector2i y_limits = box2d.getYLimits();
  // Eigen MatrixX is not bounds-checked in release; clamp to image extents so
  // an oversized or misaligned bbox can't read past the depth image buffer.
  const int img_rows = static_cast<int>(alignedDepthImg_.rows());
  const int img_cols = static_cast<int>(alignedDepthImg_.cols());
  const int row_lo = std::max(0, y_limits(0));
  const int row_hi = std::min(img_rows - 1, y_limits(1));
  const int col_lo = std::max(0, x_limits(0));
  const int col_hi = std::min(img_cols - 1, x_limits(1));
  if (row_hi < row_lo || col_hi < col_lo) {
    LOG_WARNING("2D bounding box at ", box2d.top_corner.x(), ", ",
                box2d.top_corner.y(),
                " is fully outside the depth image (", img_cols, "x", img_rows,
                ")");
    return std::nullopt;
  }
  float depth_meters;
  // All depth values in the 2D box within the range of interest
  std::vector<float> depth_values;
  for (int row_idx = row_lo; row_idx <= row_hi; ++row_idx) {
    for (int col_idx = col_lo; col_idx <= col_hi; ++col_idx) {
      depth_meters =
          alignedDepthImg_(row_idx, col_idx) * depthConversionFactor_;
      if (depth_meters <= maxDepth_ && depth_meters >= minDepth_) {
        depth_values.push_back(depth_meters);
      }
    }
  }
  if (depth_values.size() <= 1) {
    LOG_WARNING("Could not get any depth values for 2D bounding box at ",
                box2d.top_corner.x(), ", ", box2d.top_corner.y());
    return std::nullopt;
  }
  float medianDepth, madDepth;
  calculateMAD(depth_values, medianDepth, madDepth);

  // Get min and max depth
  float minimum_d = maxDepth_, maximum_d = minDepth_;
  for (auto depth : depth_values) {
    if ((depth < minimum_d) && (depth >= medianDepth - 1.5 * madDepth)) {
      minimum_d = depth;
    }
    if ((depth > maximum_d) && (depth <= medianDepth + 1.5 * madDepth)) {
      maximum_d = depth;
    }
  }

  // NOTE: Pinhole projection gives coordinates in the optical frame
  // (x_opt: right, y_opt: down, z_opt: forward). Convert to the body-aligned
  // camera frame (x: forward, y: left, z: up) so that camera_in_body_tf_ can be
  // expressed as the physical sensor pose in the body frame.
  const float x_opt =
      (box2d.top_corner.x() + 0.5f * box2d.size.x() - cx_) * medianDepth / fx_;
  const float y_opt =
      (box2d.top_corner.y() + 0.5f * box2d.size.y() - cy_) * medianDepth / fy_;
  const float z_opt = medianDepth;

  Eigen::Vector3f center_in_camera_frame, size_camera_frame;
  center_in_camera_frame(0) = z_opt;
  center_in_camera_frame(1) = -x_opt;
  center_in_camera_frame(2) = -y_opt;

  LOG_DEBUG("Median depth = ", medianDepth, ", min=", minimum_d,
            ", max=", maximum_d);

  // Size in meters, also expressed in the body-aligned camera frame
  const float size_x_opt = box2d.size.x() * medianDepth / fx_;
  const float size_y_opt = box2d.size.y() * medianDepth / fy_;
  size_camera_frame(0) = maximum_d - minimum_d;
  size_camera_frame(1) = size_x_opt;
  size_camera_frame(2) = size_y_opt;

  Eigen::Isometry3f camera_in_world_tf = body_in_world_tf_ * camera_in_body_tf_;
  // Register center in the world frame
  box3d.center = camera_in_world_tf * center_in_camera_frame;

  LOG_DEBUG("Got detected box in 3D coordinates at :", box3d.center.x(), ", ",
            box3d.center.y(), ", ", box3d.center.z());

  // Transform size from camera frame to world frame
  Eigen::Matrix3f abs_rotation = camera_in_world_tf.linear().cwiseAbs();
  box3d.size = abs_rotation * size_camera_frame;

  return box3d;
}

std::optional<Bbox3D>
DepthDetector::convertPOIto3Dbox(const PointsOfInterest &poi) {
  Bbox2D box2d(poi);
  return convert2Dboxto3Dbox(box2d);
}

float DepthDetector::getMedian(const std::vector<float> &values) {
  auto sorted_value = values;
  std::sort(sorted_value.begin(), sorted_value.end());
  const auto n = sorted_value.size();
  if (n % 2 == 0) {
    return 0.5f * (sorted_value[n / 2 - 1] + sorted_value[n / 2]);
  }
  return sorted_value[n / 2];
}

void DepthDetector::calculateMAD(const std::vector<float> &depthValues,
                                 float &median, float &mad) {
  median = getMedian(depthValues);

  std::vector<float> deviations;
  for (size_t i = 0; i < depthValues.size(); ++i) {
    deviations.push_back(std::abs(depthValues[i] - median));
  }
  mad = getMedian(deviations);
}
} // namespace Kompass
