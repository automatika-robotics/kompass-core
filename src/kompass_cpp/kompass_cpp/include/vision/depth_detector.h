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

  DepthDetector(
      const Eigen::Vector2f &depth_range,
      const Eigen::Vector3f &camera_in_body_translation,
      const Eigen::Quaternionf &camera_in_body_rotation,
      const Eigen::Vector2f &focal_length,
      const Eigen::Vector2f &principal_point,
      const float depth_conversion_factor = 1e-3,
      const CameraFrameConvention convention = CameraFrameConvention::Optical);

  DepthDetector(
      const Eigen::Vector2f &depth_range,
      const Eigen::Isometry3f &camera_in_body_tf,
      const Eigen::Vector2f &focal_length,
      const Eigen::Vector2f &principal_point,
      const float depth_conversion_factor = 1e-3,
      const CameraFrameConvention convention = CameraFrameConvention::Optical);

  /// REP 103 rotation taking optical axes to body-aligned ones.
  static const Eigen::Quaternionf &opticalToBodyAligned();

  /**
   * @brief Lifts 2D detection boxes to 3D boxes using an aligned depth image.
   *
   * For every box, the in-range depth pixels of its region are reduced to a
   * median depth and a MAD-clipped depth extent, the box centre pixel is
   * back-projected through the intrinsics at that depth, and the result is
   * moved into the world frame through the camera pose and the robot state.
   * Boxes with fewer than two in-range pixels are dropped.
   *
   * @param aligned_depth_img Non-owning zero-copy view of the depth image,
   * aligned with the image the boxes were detected in. UINT16 pixels are
   * scaled by the configured depth_conversion_factor (millimetres to metres
   * by default); FLOAT32 pixels are taken as metres. Non-finite pixels are
   * rejected by the min/max range gate.
   * @param detections 2D boxes in pixel coordinates of that image
   * @param robot_state Robot pose in the world frame. When given, the boxes
   * are returned in the world frame, otherwise in the robot body frame.
   */
  void
  updateBoxes(const DepthImageView &aligned_depth_img,
              const std::vector<Bbox2D> &detections,
              const std::optional<Path::State> &robot_state = std::nullopt);

  /**
   * @brief Lifts a set of points of interest to one 3D box using an aligned
   * depth image.
   *
   * The points are first reduced to a single 2D box (see the Bbox2D
   * constructor taking a PointsOfInterest), which then follows the same path
   * as a detection box. The result holds at most one box.
   *
   * @param aligned_depth_img See updateBoxes()
   * @param pois Points of interest in pixel coordinates of that image
   * @param robot_state See updateBoxes()
   */
  void updatePOIs(const DepthImageView &aligned_depth_img,
                  const PointsOfInterest &pois,
                  const std::optional<Path::State> &robot_state = std::nullopt);

  /**
   * @brief Lifts 2D detection boxes to 3D boxes using a point cloud.
   *
   * Every point is transformed into the camera and projected through the
   * intrinsics; the points that land inside a box contribute their depth
   * along the camera axis, and the box is then reduced exactly like the
   * depth-image variant. Points behind the camera, outside the min/max range
   * or non-finite are ignored.
   *
   * Points that project into a box but lie behind the object play the same role
   * as background pixels in a depth image. The median/MAD reduction keeps the
   * majority and clips the outliers.
   *
   * TODO: organized clouds (height > 1) currently take this projection path
   * too. A pixel-indexed path could read point (row, col) directly.
   *
   * @param cloud Non-owning zero-copy view of the PointCloud2 buffer, with
   * the points in the sensor's own frame as published. The sensor's mount
   * pose and field encoding come from setPointCloudSensor().
   * @param detections 2D boxes in pixel coordinates of the camera image
   * @param robot_state See the depth-image updateBoxes()
   */
  void
  updateBoxes(const PointCloudView &cloud,
              const std::vector<Bbox2D> &detections,
              const std::optional<Path::State> &robot_state = std::nullopt);

  /**
   * @brief Lifts a set of points of interest to one 3D box using a point
   * cloud.
   *
   * The points are first reduced to a single 2D box (see the Bbox2D
   * constructor taking a PointsOfInterest), which then follows the same path
   * as a detection box. The result holds at most one box.
   *
   * @param cloud See the point-cloud updateBoxes()
   * @param pois Points of interest in pixel coordinates of the camera image
   * @param robot_state See the depth-image updateBoxes()
   */
  void updatePOIs(const PointCloudView &cloud, const PointsOfInterest &pois,
                  const std::optional<Path::State> &robot_state = std::nullopt);

  /**
   * @brief Describes the point-cloud sensor: its mount pose in the robot body
   * frame and the encoding of its x/y/z fields.
   *
   * The mount pose is taken in the axes the cloud's frame_id names. The default
   * SensorConfig is an identity mount with FLOAT32 fields, which is what
   * applies until this is called.
   *
   * @param sensor Mount pose and field encoding of the point-cloud sensor
   */
  void setPointCloudSensor(const SensorConfig &sensor);

  /**
   * @brief Result of the last update.
   *
   * @return The lifted 3D boxes, empty when no box converted
   */
  const std::vector<Bbox3D> &get3dDetections() const { return boxes_; }

private:
  float cx_, cy_, fx_, fy_; // Depth Image camera intrinsics
  float minDepth_, maxDepth_, depthConversionFactor_;
  Eigen::Isometry3f camera_in_body_tf_, body_in_world_tf_;
  // Point-cloud sensor (mount pose + field encoding)
  SensorConfig cloud_sensor_;
  // camera_in_body_tf_^-1 * mount pose: takes a cloud point straight into the
  // body-aligned camera axes the projection works in
  Eigen::Isometry3f cloud_in_camera_tf_;

  // Output buffer storing lifted 3D boxes of last update
  std::vector<Bbox3D> boxes_;

  // Per-box scratch, refilled for every converted box. Keep the
  // allocated capacity, so it grows to the largest box seen.
  std::vector<float> depth_values_; // in-range depth values (metres)
  std::vector<float> mad_scratch_;  // absolute deviations from the median

  // Cloud path scratch. One depth-sample bucket per box of the current
  // update plus each box's inclusive pixel limits (x0, x1, y0, y1)
  std::vector<std::vector<float>> cloud_samples_;
  std::vector<Eigen::Vector4i> box_limits_;

  std::optional<Bbox3D> convert2Dboxto3Dbox(const DepthImageView &depth,
                                            const Bbox2D &box2d);

  std::optional<Bbox3D> convertPOIto3Dbox(const DepthImageView &depth,
                                          const PointsOfInterest &poi);

  /// Fills depth_values_ with the in-range depths of the box region.
  void gatherDepthSamples(const DepthImageView &depth, const Bbox2D &box2d);

  /// Projects the whole cloud once and fills cloud_samples_[i] with the
  /// in-range depths of the points that land inside boxes[i].
  void projectCloud(const PointCloudView &cloud,
                    const std::vector<Bbox2D> &boxes);

  /// Median/MAD reduction of the depth samples of one box into a Bbox3D in
  /// the world frame (body frame when no robot state was given). Shared by
  /// the depth-image and the point-cloud paths.
  std::optional<Bbox3D> boxFromDepthSamples(const Bbox2D &box2d,
                                            std::vector<float> &samples);

  void calculateMAD(std::vector<float> &depthValues, float &median, float &mad);

  /// Median via nth_element selection (O(n)).
  /// for even n the result averages the two middle order statistics
  static float getMedian(std::vector<float> &values);
};

} // namespace Kompass
