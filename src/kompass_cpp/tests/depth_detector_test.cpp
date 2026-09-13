#include "datatypes/sensors.h"
#include "datatypes/tracking.h"
#include "utils/transformation.h"
#include "vision/depth_detector.h"
#define BOOST_TEST_MODULE KOMPASS TESTS
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace Kompass;

namespace {

// The VGA pinhole the Python fixtures use
constexpr float kFx = 525.0f, kFy = 525.0f, kCx = 320.0f, kCy = 240.0f;
constexpr int kCols = 640, kRows = 480;
const Eigen::Vector2f kDepthRange{0.1f, 5.0f};
constexpr float kDepthConversion = 1e-3f;

// PointCloud2-style float32 xyz records with 4 bytes of padding, the layout
// real LiDAR drivers publish (point_step 16, offsets 0/4/8)
constexpr int kPointStep = 16;

struct PackedCloud {
  std::vector<uint8_t> data;
  int height = 1;
  int width = 0;
  int row_step = 0;

  PointCloudView view() const {
    return PointCloudView{ByteSpan(data.data(), data.size()),
                          kPointStep,
                          row_step,
                          height,
                          width,
                          0,
                          4,
                          8};
  }
};

// Packs the points into a raw buffer. With height > 1 the cloud is organized
// and every row gets `row_padding` trailing bytes, as PointCloud2 allows.
PackedCloud pack(const std::vector<Eigen::Vector3f> &points, int height = 1,
                 int row_padding = 0) {
  PackedCloud cloud;
  cloud.height = height;
  cloud.width = static_cast<int>(points.size()) / height;
  cloud.row_step = cloud.width * kPointStep + row_padding;
  cloud.data.assign(static_cast<std::size_t>(height) * cloud.row_step, 0);
  for (std::size_t i = 0; i < points.size(); ++i) {
    const int row = static_cast<int>(i) / cloud.width;
    const int col = static_cast<int>(i) % cloud.width;
    std::memcpy(&cloud.data[row * cloud.row_step + col * kPointStep],
                points[i].data(), 3 * sizeof(float));
  }
  return cloud;
}

// Pose of a camera's OPTICAL frame in the body frame: at `translation`,
// looking along body +x turned by `yaw`
Eigen::Isometry3f opticalCameraPose(const Eigen::Vector3f &translation,
                                    const float yaw) {
  const Eigen::Quaternionf rotation =
      Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
      DepthDetector::opticalToBodyAligned();
  return getTransformation(rotation, translation);
}

DepthDetector makeDetector(const Eigen::Isometry3f &camera_optical_in_body) {
  return DepthDetector(kDepthRange, camera_optical_in_body,
                       Eigen::Vector2f{kFx, kFy}, Eigen::Vector2f{kCx, kCy},
                       kDepthConversion,
                       DepthDetector::CameraFrameConvention::Optical);
}

// A SensorConfig (mount pose in the body frame, FLOAT32 fields) from an
// isometry, the way kompass builds one from a TF lookup
SensorConfig sensorAt(const Eigen::Isometry3f &sensor_in_body) {
  SensorConfig sensor;
  sensor.position = sensor_in_body.translation();
  const Eigen::Quaternionf q(sensor_in_body.linear());
  sensor.rotation = Eigen::Vector4f{q.x(), q.y(), q.z(), q.w()};
  return sensor;
}

// A synthetic uint16 depth image (millimetres) and the same scene as an
// optical-frame point cloud: every pixel back-projected at its centre
struct Scene {
  std::vector<uint16_t> depth_mm;
  std::vector<Eigen::Vector3f> points;

  DepthImageView depthView() const {
    return DepthImageView(
        ByteSpan(reinterpret_cast<const uint8_t *>(depth_mm.data()),
                 depth_mm.size() * sizeof(uint16_t)),
        kRows, kCols, PointFieldType::UINT16);
  }
};

// A 4.5 m background with one plateau per box that also ramps a little along
// the columns, so the median and the MAD extent are both non-trivial
Scene makeScene(const std::vector<std::pair<Bbox2D, uint16_t>> &plateaus) {
  Scene scene;
  scene.depth_mm.assign(static_cast<std::size_t>(kRows) * kCols, 4500);
  for (const auto &[box, depth] : plateaus) {
    const Eigen::Vector2i x_limits = box.getXLimits();
    const Eigen::Vector2i y_limits = box.getYLimits();
    for (int row = y_limits(0); row <= y_limits(1); ++row) {
      for (int col = x_limits(0); col <= x_limits(1); ++col) {
        scene.depth_mm[row * kCols + col] =
            static_cast<uint16_t>(depth + 2 * (col - x_limits(0)));
      }
    }
  }
  scene.points.reserve(scene.depth_mm.size());
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float d = scene.depth_mm[row * kCols + col] * kDepthConversion;
      scene.points.emplace_back((col + 0.5f - kCx) * d / kFx,
                                (row + 0.5f - kCy) * d / kFy, d);
    }
  }
  return scene;
}

void checkSameBoxes(const std::vector<Bbox3D> &from_depth,
                    const std::vector<Bbox3D> &from_cloud) {
  BOOST_REQUIRE_EQUAL(from_depth.size(), from_cloud.size());
  for (std::size_t i = 0; i < from_depth.size(); ++i) {
    BOOST_CHECK_SMALL((from_depth[i].center - from_cloud[i].center).norm(),
                      1e-3f);
    BOOST_CHECK_SMALL((from_depth[i].size - from_cloud[i].size).norm(), 1e-3f);
    BOOST_CHECK(from_depth[i].center_img_frame == from_cloud[i].center_img_frame);
    BOOST_CHECK(from_depth[i].size_img_frame == from_cloud[i].size_img_frame);
    BOOST_CHECK(from_cloud[i].pc_points.empty());
  }
}

} // namespace

// The projection path must reproduce the depth-image path: a cloud made by
// back-projecting the image, handed over in the camera's optical frame, gives
// the same 3D boxes (two boxes at once, with a robot pose so the result is in
// the world frame)
BOOST_AUTO_TEST_CASE(cloud_path_matches_depth_image_path) {
  const Bbox2D box_a(Eigen::Vector2i{270, 190}, Eigen::Vector2i{100, 100});
  const Bbox2D box_b(Eigen::Vector2i{50, 300}, Eigen::Vector2i{60, 40});
  const Scene scene = makeScene({{box_a, 2000}, {box_b, 3000}});
  const Path::State robot_state(1.0, -2.0, 0.3, 0.0);
  const Eigen::Isometry3f camera =
      opticalCameraPose(Eigen::Vector3f{0.2f, 0.05f, 0.6f}, 0.17f);
  const std::vector<Bbox2D> boxes{box_a, box_b};

  DepthDetector from_depth = makeDetector(camera);
  from_depth.updateBoxes(scene.depthView(), boxes, robot_state);
  BOOST_REQUIRE_EQUAL(from_depth.get3dDetections().size(), 2);

  DepthDetector from_cloud = makeDetector(camera);
  // The cloud is expressed in the optical frame, so its sensor sits at the
  // camera's optical pose
  from_cloud.setPointCloudSensor(sensorAt(camera));
  const PackedCloud cloud = pack(scene.points);
  from_cloud.updateBoxes(cloud.view(), boxes,
                         robot_state);

  checkSameBoxes(from_depth.get3dDetections(), from_cloud.get3dDetections());
}

// Same scene as an organized cloud with padded rows: the row padding must be
// skipped, not decoded as points
BOOST_AUTO_TEST_CASE(cloud_path_handles_organized_row_padding) {
  const Bbox2D box(Eigen::Vector2i{270, 190}, Eigen::Vector2i{100, 100});
  const Scene scene = makeScene({{box, 2000}});
  const Eigen::Isometry3f camera =
      opticalCameraPose(Eigen::Vector3f{0.2f, 0.0f, 0.6f}, 0.0f);

  DepthDetector from_depth = makeDetector(camera);
  from_depth.updateBoxes(scene.depthView(), {box});

  DepthDetector from_cloud = makeDetector(camera);
  from_cloud.setPointCloudSensor(sensorAt(camera));
  const PackedCloud cloud = pack(scene.points, kRows, 8);
  from_cloud.updateBoxes(cloud.view(), {box});

  checkSameBoxes(from_depth.get3dDetections(), from_cloud.get3dDetections());
}

// A LiDAR mounted away from the camera, in its own body-aligned frame: a
// cluster at a known body-frame location comes back at that location, while
// points behind the camera, out of range (a wall behind the cluster that
// projects into the same box), outside the box, or non-finite are ignored
BOOST_AUTO_TEST_CASE(cloud_path_lifts_lidar_cluster_through_mount_pose) {
  const Eigen::Isometry3f camera =
      opticalCameraPose(Eigen::Vector3f{0.3f, 0.0f, 0.5f}, 0.0f);
  DepthDetector detector = makeDetector(camera);

  const Eigen::Isometry3f lidar_in_body = getTransformation(
      Eigen::Quaternionf(Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitZ())),
      Eigen::Vector3f{-0.1f, 0.0f, 0.8f});
  detector.setPointCloudSensor(sensorAt(lidar_in_body));
  const Eigen::Isometry3f body_in_lidar = lidar_in_body.inverse();

  // Everything is placed in body coordinates and then moved into the LiDAR
  // frame, which is what the sensor publishes
  std::vector<Eigen::Vector3f> points;
  const Eigen::Vector3f cluster{3.0f, 0.5f, 0.4f};
  for (int i = -5; i <= 5; ++i) {
    for (int j = -5; j <= 5; ++j) {
      for (int k = -1; k <= 1; ++k) {
        points.push_back(
            body_in_lidar *
            (cluster + Eigen::Vector3f{0.05f * k, 0.04f * i, 0.04f * j}));
      }
    }
  }
  const std::size_t cluster_points = points.size();
  // A wall 5.7 m from the camera, more points than the cluster: if the range
  // gate failed it would take over the median
  for (int i = -10; i <= 10; ++i) {
    for (int j = -10; j <= 10; ++j) {
      points.push_back(body_in_lidar *
                       Eigen::Vector3f{6.0f, 0.5f + 0.02f * i, 0.4f + 0.02f * j});
    }
  }
  BOOST_REQUIRE_GT(points.size() - cluster_points, cluster_points);
  points.push_back(body_in_lidar * Eigen::Vector3f{-2.0f, 0.5f, 0.4f});
  points.push_back(body_in_lidar * Eigen::Vector3f{3.0f, -1.5f, 0.4f});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  points.emplace_back(nan, nan, nan);

  // The cluster's projection spans u in [184, 262] and v in [221, 298]
  const Bbox2D box(Eigen::Vector2i{180, 218}, Eigen::Vector2i{86, 84});
  const PackedCloud cloud = pack(points);
  detector.updateBoxes(cloud.view(), {box});

  const auto &boxes = detector.get3dDetections();
  BOOST_REQUIRE_EQUAL(boxes.size(), 1);
  // Body frame since no robot state was given. The centre is the box centre
  // pixel at the median depth, which sits within a few millimetres of the
  // cluster centre for this box
  BOOST_CHECK_SMALL((boxes[0].center - cluster).norm(), 0.02f);
  // Depth extent comes from the +-5 cm spread, MAD-clipped
  BOOST_CHECK_GT(boxes[0].size.x(), 0.0f);
  BOOST_CHECK_LT(boxes[0].size.x(), 0.2f);
}

// Nothing to lift: no point lands in the box, an empty view, or a view with
// no x/y/z offsets all give an empty result without touching the buffer
BOOST_AUTO_TEST_CASE(cloud_path_empty_results) {
  const Eigen::Isometry3f camera =
      opticalCameraPose(Eigen::Vector3f{0.0f, 0.0f, 0.0f}, 0.0f);
  DepthDetector detector = makeDetector(camera);
  const Bbox2D box(Eigen::Vector2i{300, 220}, Eigen::Vector2i{40, 40});

  // In view, in range, but projecting far from the box
  const PackedCloud outside = pack({Eigen::Vector3f{2.0f, 1.5f, 0.0f},
                                    Eigen::Vector3f{2.0f, -1.5f, 0.0f}});
  detector.updateBoxes(outside.view(), {box});
  BOOST_CHECK(detector.get3dDetections().empty());

  detector.updateBoxes(PointCloudView{}, {box});
  BOOST_CHECK(detector.get3dDetections().empty());

  PointCloudView no_offsets = outside.view();
  no_offsets.z_offset = -1;
  detector.updateBoxes(no_offsets, {box});
  BOOST_CHECK(detector.get3dDetections().empty());

  detector.updateBoxes(outside.view(), {});
  BOOST_CHECK(detector.get3dDetections().empty());
}

// Points of interest reduce to one box and take the same path
BOOST_AUTO_TEST_CASE(cloud_path_matches_depth_image_path_for_pois) {
  const Bbox2D region(Eigen::Vector2i{270, 190}, Eigen::Vector2i{100, 100});
  const Scene scene = makeScene({{region, 2000}});
  const Eigen::Isometry3f camera =
      opticalCameraPose(Eigen::Vector3f{0.2f, 0.0f, 0.6f}, -0.1f);
  const PointsOfInterest poi(
      {Eigen::Vector2i{300, 220}, Eigen::Vector2i{340, 260},
       Eigen::Vector2i{320, 240}, Eigen::Vector2i{310, 250},
       Eigen::Vector2i{330, 230}},
      Eigen::Vector2i{kCols, kRows});
  const Path::State robot_state(0.5, 0.25, -0.4, 0.0);

  DepthDetector from_depth = makeDetector(camera);
  from_depth.updatePOIs(scene.depthView(), poi, robot_state);
  BOOST_REQUIRE_EQUAL(from_depth.get3dDetections().size(), 1);

  DepthDetector from_cloud = makeDetector(camera);
  from_cloud.setPointCloudSensor(sensorAt(camera));
  const PackedCloud cloud = pack(scene.points);
  from_cloud.updatePOIs(cloud.view(), poi,
                        robot_state);

  checkSameBoxes(from_depth.get3dDetections(), from_cloud.get3dDetections());
}

// Every lifted box says which input it came from and how many depth readings
// it rests on, on both paths.
BOOST_AUTO_TEST_CASE(lifted_boxes_carry_their_provenance) {
  const Bbox3D untouched;
  BOOST_CHECK_EQUAL(untouched.sample_count, 0);
  BOOST_CHECK_EQUAL(untouched.source_index, -1);

  // box_a rests on a plateau in range; box_b sits beyond the depth range and
  // is dropped
  const Bbox2D box_a(Eigen::Vector2i{270, 190}, Eigen::Vector2i{100, 100});
  const Bbox2D box_b(Eigen::Vector2i{50, 300}, Eigen::Vector2i{60, 40});
  const Scene scene = makeScene({{box_a, 2000}, {box_b, 6000}});
  const std::vector<Bbox2D> boxes{box_b, box_a};
  const Eigen::Isometry3f camera =
      opticalCameraPose(Eigen::Vector3f{0.0f, 0.0f, 0.0f}, 0.0f);

  // The readings the image path can use: in-range pixels within the box's
  // inclusive limits
  int usable = 0;
  const Eigen::Vector2i x_limits = box_a.getXLimits();
  const Eigen::Vector2i y_limits = box_a.getYLimits();
  for (int row = y_limits(0); row <= y_limits(1); ++row) {
    for (int col = x_limits(0); col <= x_limits(1); ++col) {
      const float metres = scene.depth_mm[row * kCols + col] * kDepthConversion;
      usable += (metres >= kDepthRange(0) && metres <= kDepthRange(1)) ? 1 : 0;
    }
  }

  DepthDetector from_depth = makeDetector(camera);
  from_depth.updateBoxes(scene.depthView(), boxes);
  BOOST_REQUIRE_EQUAL(from_depth.get3dDetections().size(), 1);
  const Bbox3D &from_image = from_depth.get3dDetections()[0];
  BOOST_CHECK_EQUAL(from_image.source_index, 1);
  BOOST_CHECK_EQUAL(from_image.sample_count, usable);

  // One point per pixel, so the cloud path rests on the same readings
  DepthDetector from_cloud = makeDetector(camera);
  from_cloud.setPointCloudSensor(sensorAt(camera));
  const PackedCloud cloud = pack(scene.points);
  from_cloud.updateBoxes(cloud.view(), boxes);
  BOOST_REQUIRE_EQUAL(from_cloud.get3dDetections().size(), 1);
  BOOST_CHECK_EQUAL(from_cloud.get3dDetections()[0].source_index, 1);
  BOOST_CHECK_EQUAL(from_cloud.get3dDetections()[0].sample_count, usable);

  // A points-of-interest set is one input
  const PointsOfInterest poi(
      {Eigen::Vector2i{300, 220}, Eigen::Vector2i{340, 260},
       Eigen::Vector2i{320, 240}},
      Eigen::Vector2i{kCols, kRows});
  from_depth.updatePOIs(scene.depthView(), poi);
  BOOST_REQUIRE_EQUAL(from_depth.get3dDetections().size(), 1);
  BOOST_CHECK_EQUAL(from_depth.get3dDetections()[0].source_index, 0);
  BOOST_CHECK_GT(from_depth.get3dDetections()[0].sample_count, 1);
}
