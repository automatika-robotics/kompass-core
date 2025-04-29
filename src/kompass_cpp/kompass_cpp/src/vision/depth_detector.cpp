#include "vision/depth_detector.h"
#include "datatypes/tracking.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <memory>
#include <optional>
#include <vector>

namespace Kompass {

DepthDetector::DepthDetector(const Eigen::Vector2f &depth_range,
                             const Eigen::Vector3f &body_to_camera_translation,
                             const Eigen::Quaternionf &body_to_camera_rotation,
                             const Eigen::Vector2f &focal_length,
                             const Eigen::Vector2f &principal_point,
                             const float depth_conversion_factor) {
  // Set camera tf
  auto body_to_camera_tf =
      getTransformation(body_to_camera_rotation, body_to_camera_translation);
  DepthDetector(depth_range, body_to_camera_tf, focal_length, principal_point,
                depth_conversion_factor);
}

DepthDetector::DepthDetector(
    const Eigen::Vector2f &depth_range,
    const Eigen::Isometry3f &body_to_camera_tf,
    const Eigen::Vector2f &focal_length, const Eigen::Vector2f &principal_point,
    const float depth_conversion_factor) { // Range of interest for depth values
                                           // in meters
  minDepth_ = depth_range(0);
  maxDepth_ = depth_range(1);
  // Factor to convert depth image data to meters (in ROS2 its given in mm ->
  // depthConversionFactor = 1e-3)
  depthConversionFactor_ = depth_conversion_factor;
  // Set camera tf
  body_to_camera_tf_ = body_to_camera_tf;
  world_to_body_tf_ = Eigen::Isometry3f::Identity();

  // Set camera  intrinsic parameters
  fx_ = focal_length.x();
  fy_ = focal_length.y();
  cx_ = principal_point.x();
  cy_ = principal_point.y();
}

std::optional<std::vector<Bbox3D>> DepthDetector::get3dDetections() const {
  if (boxes_) {
    return *boxes_;
  }
  return std::nullopt;
}

void DepthDetector::updateState(const Path::State &current_state) {

  world_to_body_tf_ = getTransformation(current_state);
}

void DepthDetector::updateState(const Eigen::Isometry3f &robot_tf) {
  world_to_body_tf_ = robot_tf;
}

void DepthDetector::updateBoxes(const Eigen::MatrixXi aligned_depth_img,
                                const std::vector<Bbox2D> &detections) {
  alignedDepthImg_ = aligned_depth_img;
  boxes_ = std::make_unique<std::vector<Bbox3D>>();
  for (auto box2d : detections) {
    auto converted_box = convert2Dboxto3Dbox(box2d);
    if (converted_box) {
      boxes_->push_back(converted_box.value());
    }
  }
}

std::optional<Bbox3D> DepthDetector::convert2Dboxto3Dbox(const Bbox2D &box2d) {
  Bbox3D box3d(box2d);
  Eigen::Vector2i x_limits = box2d.getXLimits();
  Eigen::Vector2i y_limits = box2d.getYLimits();
  float depth_meters;
  // All depth values in the 2D box within the range of interest
  std::vector<float> depth_values;
  for (int row_idx = y_limits(0); row_idx <= y_limits(1); ++row_idx) {
    for (int col_idx = x_limits(0); col_idx <= x_limits(1); ++col_idx) {
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

  // Convert from 2D box center in the pixel frame to the 3D box center in the
  // camera frame
  Eigen::Vector3f center_img_frame, size_img_frame;
  LOG_DEBUG("Got detected box in 2D pixel frame at :", box2d.top_corner.x(),
           ", ", box2d.top_corner.y(), ", size = ", box2d.size.x(), ", ",
           box2d.size.y());
  center_img_frame(0) = (box2d.top_corner.x() + 0.5 * box2d.size.x() - cx_) *
                        medianDepth / this->fx_;
  center_img_frame(1) = (box2d.top_corner.y() + 0.5 * box2d.size.y() - cy_) *
                        medianDepth / this->fy_;
  center_img_frame(2) = medianDepth;

  LOG_DEBUG("Got detected box in 3D camera frame at :", center_img_frame.x(),
           ", ", center_img_frame.y(), ", ", center_img_frame.z());

  LOG_DEBUG("Median depth = ", medianDepth);

  // Size in meters
  size_img_frame(0) = box2d.size.x() * medianDepth / this->fx_;
  size_img_frame(1) = box2d.size.y() * medianDepth / this->fy_;
  size_img_frame(2) = maximum_d - minimum_d;

  // Register center in the world frame
  box3d.center = world_to_body_tf_ * body_to_camera_tf_ * center_img_frame;

  // Transform size from camera frame to world frame
  box3d.size(0) = (world_to_body_tf_ * body_to_camera_tf_ *
                   Eigen::Vector3f(size_img_frame(0), 0, 0))
                      .norm();
  box3d.size(1) = (world_to_body_tf_ * body_to_camera_tf_ *
                   Eigen::Vector3f(0, size_img_frame(1), 0))
                      .norm();
  box3d.size(2) = (world_to_body_tf_ * body_to_camera_tf_ *
                   Eigen::Vector3f(0, 0, size_img_frame(2)))
                      .norm();

  return box3d;
}

float DepthDetector::getMedian(const std::vector<float> &values) {
  float median;
  auto sorted_value = values;
  std::sort(sorted_value.begin(), sorted_value.end());
  // number of items
  auto num = sorted_value.size() - 1;
  if (num % 2 == 0) {
    median = (sorted_value[num / 2] + sorted_value[num / 2 + 1]) / 2;
  } else {
    median = sorted_value[(num + 1) / 2];
  }
  return median;
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
