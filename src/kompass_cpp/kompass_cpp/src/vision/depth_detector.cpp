#include "vision/depth_detector.h"
#include "datatypes/tracking.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <optional>
#include <vector>

namespace Kompass {

const Eigen::Quaternionf &DepthDetector::opticalToBodyAligned() {
  // Takes (x right, y down, z forward) to (x forward, y left, z up).
  // Eigen's quaternion constructor is (w, x, y, z).
  static const Eigen::Quaternionf kOpticalToBody{0.5f, -0.5f, 0.5f, -0.5f};
  return kOpticalToBody;
}

DepthDetector::DepthDetector(const Eigen::Vector2f &depth_range,
                             const Eigen::Vector3f &camera_in_body_translation,
                             const Eigen::Quaternionf &camera_in_body_rotation,
                             const Eigen::Vector2f &focal_length,
                             const Eigen::Vector2f &principal_point,
                             const float depth_conversion_factor,
                             const CameraFrameConvention convention)
    : DepthDetector(depth_range,
                    getTransformation(camera_in_body_rotation,
                                      camera_in_body_translation),
                    focal_length, principal_point, depth_conversion_factor,
                    convention) {}

DepthDetector::DepthDetector(
    const Eigen::Vector2f &depth_range,
    const Eigen::Isometry3f &camera_in_body_tf,
    const Eigen::Vector2f &focal_length, const Eigen::Vector2f &principal_point,
    const float depth_conversion_factor,
    const CameraFrameConvention convention) { // Range of interest for depth
                                              // values in meters
  minDepth_ = depth_range(0);
  maxDepth_ = depth_range(1);
  // Factor to convert depth image data to meters (in ROS2 its given in mm ->
  // depthConversionFactor = 1e-3)
  depthConversionFactor_ = depth_conversion_factor;
  // Set camera tf.
  //
  // Projection below turns the optical axes into body-aligned ones before
  // applying this transform, so a pose given in the optical convention has
  // that fixed quarter rotation taken back out here. Without this the
  // rotation lands twice and every detection is 90 degrees off.
  camera_in_body_tf_ =
      convention == CameraFrameConvention::Optical
          ? camera_in_body_tf *
                Eigen::Isometry3f(opticalToBodyAligned().conjugate())
          : camera_in_body_tf;

  // Set camera  intrinsic parameters
  fx_ = focal_length.x();
  fy_ = focal_length.y();
  cx_ = principal_point.x();
  cy_ = principal_point.y();

  body_in_world_tf_ = Eigen::Isometry3f::Identity();
  // Until a sensor is described, a point cloud is taken as an identity mount
  // with FLOAT32 fields (SensorConfig defaults)
  setPointCloudSensor(SensorConfig{});
}

void DepthDetector::setPointCloudSensor(const SensorConfig &sensor) {
  cloud_sensor_ = sensor;
  cloud_in_camera_tf_ = camera_in_body_tf_.inverse() * sensor.tfBody();
}

void DepthDetector::updateBoxes(const DepthImageView &aligned_depth_img,
                                const std::vector<Bbox2D> &detections,
                                const std::optional<Path::State> &robot_state) {
  if (robot_state.has_value()) {
    body_in_world_tf_ = getTransformation(robot_state.value());
  }
  boxes_.clear();
  for (std::size_t i = 0; i < detections.size(); ++i) {
    auto converted_box = convert2Dboxto3Dbox(aligned_depth_img, detections[i]);
    if (converted_box) {
      converted_box->source_index = static_cast<int>(i);
      boxes_.push_back(std::move(converted_box.value()));
    }
  }
}

void DepthDetector::updatePOIs(const DepthImageView &aligned_depth_img,
                               const PointsOfInterest &poi,
                               const std::optional<Path::State> &robot_state) {
  if (robot_state.has_value()) {
    body_in_world_tf_ = getTransformation(robot_state.value());
  }
  boxes_.clear();
  auto converted_box = convertPOIto3Dbox(aligned_depth_img, poi);
  if (converted_box) {
    converted_box->source_index = 0;
    boxes_.push_back(std::move(converted_box.value()));
  }
}

void DepthDetector::updateBoxes(const PointCloudView &cloud,
                                const std::vector<Bbox2D> &detections,
                                const std::optional<Path::State> &robot_state) {
  if (robot_state.has_value()) {
    body_in_world_tf_ = getTransformation(robot_state.value());
  }
  boxes_.clear();
  if (detections.empty()) {
    return;
  }
  projectCloud(cloud, detections);
  for (std::size_t i = 0; i < detections.size(); ++i) {
    auto converted_box = boxFromDepthSamples(detections[i], cloud_samples_[i]);
    if (converted_box) {
      converted_box->source_index = static_cast<int>(i);
      boxes_.push_back(std::move(converted_box.value()));
    }
  }
}

void DepthDetector::updatePOIs(const PointCloudView &cloud,
                               const PointsOfInterest &poi,
                               const std::optional<Path::State> &robot_state) {
  // A points-of-interest set reduces to one 2D box which then takes the same
  // path as a detection
  updateBoxes(cloud, {Bbox2D(poi)}, robot_state);
}

std::optional<Bbox3D>
DepthDetector::convert2Dboxto3Dbox(const DepthImageView &depth,
                                   const Bbox2D &box2d) {
  gatherDepthSamples(depth, box2d);
  return boxFromDepthSamples(box2d, depth_values_);
}

void DepthDetector::gatherDepthSamples(const DepthImageView &depth,
                                       const Bbox2D &box2d) {
  Eigen::Vector2i x_limits = box2d.getXLimits();
  Eigen::Vector2i y_limits = box2d.getYLimits();
  // FLOAT32 pixels are metres already; UINT16 scale by the configured factor
  const float to_meters = depth.field_type == PointFieldType::FLOAT32
                              ? 1.0f
                              : depthConversionFactor_;
  float depth_meters;
  // All depth values in the 2D box within the range of interest.
  // NaN padding in float images -> rejected.
  depth_values_.clear();
  depth_values_.reserve(
      static_cast<std::size_t>(y_limits(1) - y_limits(0) + 1) *
      (x_limits(1) - x_limits(0) + 1));
  for (int row_idx = y_limits(0); row_idx <= y_limits(1); ++row_idx) {
    for (int col_idx = x_limits(0); col_idx <= x_limits(1); ++col_idx) {
      depth_meters = depth.at(row_idx, col_idx) * to_meters;
      if (depth_meters <= maxDepth_ && depth_meters >= minDepth_) {
        depth_values_.push_back(depth_meters);
      }
    }
  }
}

void DepthDetector::projectCloud(const PointCloudView &cloud,
                                 const std::vector<Bbox2D> &boxes) {
  const PointFieldType field_type = cloud_sensor_.cloud_field_type;
  // TODO: an organized cloud (height > 1, height x width equal to the
  // detection image) in the camera frame maps pixel (row, col) to point
  // (row, col) directly. Such a cloud could index the box region the way
  // gatherDepthSamples does instead of projecting every point.
  cloud_samples_.resize(boxes.size());
  box_limits_.resize(boxes.size());
  // NOTE: Union rectangle of all boxes in inclusive pixel limits. A point that
  // projects outside it cannot fall inside any box, which keeps the per-box
  // loop off most of the cloud
  int u_min = INT_MAX, u_max = INT_MIN, v_min = INT_MAX, v_max = INT_MIN;
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    cloud_samples_[i].clear();
    const Eigen::Vector2i x_limits = boxes[i].getXLimits();
    const Eigen::Vector2i y_limits = boxes[i].getYLimits();
    box_limits_[i] =
        Eigen::Vector4i(x_limits(0), x_limits(1), y_limits(0), y_limits(1));
    u_min = std::min(u_min, x_limits(0));
    u_max = std::max(u_max, x_limits(1));
    v_min = std::min(v_min, y_limits(0));
    v_max = std::max(v_max, y_limits(1));
  }

  if (cloud.empty() || !cloud.offsetsValid()) {
    LOG_WARNING("Point cloud is empty or has no x/y/z fields, no depth "
                "samples for the 2D boxes");
    return;
  }
  // The layout is checked once up front instead of per point. The last point
  // of the last row must fit in the buffer with fields included
  const int elem_size = elementSizeOf(field_type);
  const std::size_t required_bytes =
      static_cast<std::size_t>(cloud.height - 1) * cloud.row_step +
      static_cast<std::size_t>(cloud.width - 1) * cloud.point_step +
      std::max({cloud.x_offset, cloud.y_offset, cloud.z_offset}) + elem_size;
  if (required_bytes > cloud.data.size()) {
    LOG_WARNING("Point cloud layout exceeds its buffer (", required_bytes,
                " bytes needed, ", cloud.data.size(),
                " given), no depth "
                "samples for the 2D boxes");
    return;
  }

  const Eigen::Matrix3f rotation = cloud_in_camera_tf_.linear();
  const Eigen::Vector3f translation = cloud_in_camera_tf_.translation();
  const uint8_t *bytes = cloud.data.data();

  // Walk the rows by their payload only (width points), so the row padding of
  // organized clouds is never decoded as points
  const int row_bytes = cloud.width * cloud.point_step;
  for (int row = 0; row < cloud.height; ++row) {
    for (int col = 0; col < row_bytes; col += cloud.point_step) {
      const std::size_t point_start = row * cloud.row_step + col;
      const Eigen::Vector3f point{
          load_and_cast_val(bytes, point_start + cloud.x_offset, field_type),
          load_and_cast_val(bytes, point_start + cloud.y_offset, field_type),
          load_and_cast_val(bytes, point_start + cloud.z_offset, field_type)};
      // Into the body-aligned camera axes: x forward, y left, z up
      const Eigen::Vector3f in_camera = rotation * point + translation;
      const float depth = in_camera.x();

      // Filter non finite points
      if (!(depth >= minDepth_ && depth <= maxDepth_) || depth <= 0.0f) {
        continue;
      }

      // NOTE: Pinhole projection: optical x (right) is -y and optical y (down)
      // is -z of the body-aligned axes. Pixel i spans [i, i + 1) so the pixel a
      // continuous coordinate lands in is its floor
      const float inv_depth = 1.0f / depth;
      const int u = static_cast<int>(
          std::floor(fx_ * (-in_camera.y()) * inv_depth + cx_));
      const int v = static_cast<int>(
          std::floor(fy_ * (-in_camera.z()) * inv_depth + cy_));
      if (u < u_min || u > u_max || v < v_min || v > v_max) {
        continue;
      }
      for (std::size_t i = 0; i < boxes.size(); ++i) {
        const Eigen::Vector4i &limits = box_limits_[i];
        if (u >= limits(0) && u <= limits(1) && v >= limits(2) &&
            v <= limits(3)) {
          cloud_samples_[i].push_back(depth);
        }
      }
    }
  }
}

std::optional<Bbox3D>
DepthDetector::boxFromDepthSamples(const Bbox2D &box2d,
                                   std::vector<float> &samples) {
  Bbox3D box3d(box2d);
  box3d.sample_count = static_cast<int>(samples.size());
  if (samples.size() <= 1) {
    LOG_WARNING("Could not get any depth values for 2D bounding box at ",
                box2d.top_corner.x(), ", ", box2d.top_corner.y());
    return std::nullopt;
  }
  float medianDepth, madDepth;
  calculateMAD(samples, medianDepth, madDepth);

  // Get min and max depth
  float minimum_d = maxDepth_, maximum_d = minDepth_;
  for (auto depth_val : samples) {
    if ((depth_val < minimum_d) &&
        (depth_val >= medianDepth - 1.5 * madDepth)) {
      minimum_d = depth_val;
    }
    if ((depth_val > maximum_d) &&
        (depth_val <= medianDepth + 1.5 * madDepth)) {
      maximum_d = depth_val;
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

  // Size in meters, also expressed in the body-aligned camera frame
  const float size_x_opt = box2d.size.x() * medianDepth / fx_;
  const float size_y_opt = box2d.size.y() * medianDepth / fy_;
  size_camera_frame(0) = maximum_d - minimum_d;
  size_camera_frame(1) = size_x_opt;
  size_camera_frame(2) = size_y_opt;

  Eigen::Isometry3f camera_in_world_tf = body_in_world_tf_ * camera_in_body_tf_;
  // Register center in the world frame
  box3d.center = camera_in_world_tf * center_in_camera_frame;

  // Transform size from camera frame to world frame
  Eigen::Matrix3f abs_rotation = camera_in_world_tf.linear().cwiseAbs();
  box3d.size = abs_rotation * size_camera_frame;

  return box3d;
}

std::optional<Bbox3D>
DepthDetector::convertPOIto3Dbox(const DepthImageView &depth,
                                 const PointsOfInterest &poi) {
  Bbox2D box2d(poi);
  return convert2Dboxto3Dbox(depth, box2d);
}

float DepthDetector::getMedian(std::vector<float> &values) {
  const auto n = values.size();
  const auto mid = values.begin() + n / 2;
  // Selection: *mid becomes the n/2-th order statistic and everything left of
  // it is <= *mid
  std::nth_element(values.begin(), mid, values.end());
  const float upper = *mid;
  if (n % 2 == 0) { // for even elements
    // The lower middle is the largest element of the left partition.
    // Equivalent to sorted[n/2 - 1]
    const float lower = *std::max_element(values.begin(), mid);
    return 0.5f * (lower + upper);
  }
  return upper; // for odd elements
}

void DepthDetector::calculateMAD(std::vector<float> &depthValues, float &median,
                                 float &mad) {
  median = getMedian(depthValues);

  // resize and fill member scratch
  mad_scratch_.resize(depthValues.size());
  for (size_t i = 0; i < depthValues.size(); ++i) {
    mad_scratch_[i] = std::abs(depthValues[i] - median);
  }
  mad = getMedian(mad_scratch_);
}
} // namespace Kompass
