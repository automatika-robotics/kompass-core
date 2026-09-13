// Parametrized fixture-based test for RGBDFollower.
//
// Loads each fixture under tests/resources/vision_follower/<case>/ which
// contains:
//   - depth.png: 16-bit single-channel depth image (millimeters)
//   - case.json: camera intrinsics, robot state, 2D detections, click pixel,
//                and loose expected bounds for the resulting control command.
//
// Mirrors tests/test_vision_follower.py so both layers exercise the same
// data. To add cases, edit tests/resources/vision_follower/generate_fixtures.py
// and re-run it (or drop a new fixture directory in by hand).

#include "controllers/rgbd_follower.h"
#include "datatypes/control.h"
#include "datatypes/tracking.h"
#include "utils/logger.h"

#define BOOST_TEST_MODULE VISION_FOLLOWER_FIXTURE_TESTS
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace Kompass;
using json = nlohmann::json;
namespace fs = boost::filesystem;

namespace {

struct FixtureCase {
  std::string name;
  fs::path dir;
};

// Discover fixtures by walking up from the executable to find the source tree
// resources. The kompass build layout drops binaries in
// build/<py-tag>/src/kompass_cpp/tests/ so we walk to the repo root and append
// tests/resources/vision_follower.
fs::path locate_fixture_root() {
  fs::path here = boost::dll::program_location().parent_path();
  for (int i = 0; i < 8; ++i) {
    fs::path candidate = here / "tests" / "resources" / "vision_follower";
    if (fs::is_directory(candidate)) {
      return candidate;
    }
    if (here.has_parent_path()) {
      here = here.parent_path();
    } else {
      break;
    }
  }
  // Fall back to the source-tree path (cmake current source dir).
  fs::path src_default = fs::path(__FILE__)
                             .parent_path()
                             .parent_path()
                             .parent_path()
                             .parent_path() /
                         "tests" / "resources" / "vision_follower";
  return src_default;
}

std::vector<FixtureCase> discover_fixtures() {
  std::vector<FixtureCase> out;
  fs::path root = locate_fixture_root();
  if (!fs::is_directory(root)) {
    LOG_ERROR("Fixture root not found at ", root.string());
    return out;
  }
  for (auto const &entry : fs::directory_iterator(root)) {
    if (!fs::is_directory(entry.path()))
      continue;
    if (!fs::exists(entry.path() / "case.json"))
      continue;
    if (!fs::exists(entry.path() / "depth.png"))
      continue;
    out.push_back({entry.path().filename().string(), entry.path()});
  }
  std::sort(out.begin(), out.end(),
            [](const FixtureCase &a, const FixtureCase &b) {
              return a.name < b.name;
            });
  return out;
}

cv::Mat load_depth_png(const fs::path &png_path) {
  cv::Mat raw = cv::imread(png_path.string(), cv::IMREAD_UNCHANGED);
  BOOST_REQUIRE_MESSAGE(!raw.empty(),
                        "Could not load depth.png at " << png_path.string());
  BOOST_REQUIRE_MESSAGE(
      raw.type() == CV_16UC1,
      "depth.png must be 16-bit single-channel: " << png_path.string());
  BOOST_REQUIRE(raw.isContinuous());
  return raw;
}

// Zero-copy view over the row-major cv::Mat buffer (16UC1 = uint16 mm).
// The Mat must outlive every use of the view
DepthImageView depth_view(const cv::Mat &img) {
  return DepthImageView(
      ByteSpan(img.ptr<uint8_t>(), img.total() * img.elemSize()), img.rows,
      img.cols, PointFieldType::UINT16);
}

std::vector<Bbox2D> parse_detections(const json &case_json) {
  std::vector<Bbox2D> dets;
  const int img_w = case_json["camera"]["img_w"].get<int>();
  const int img_h = case_json["camera"]["img_h"].get<int>();
  for (const auto &d : case_json["detections"]) {
    Eigen::Vector2i tl{d["top_left"][0].get<int>(),
                       d["top_left"][1].get<int>()};
    Eigen::Vector2i sz{d["size"][0].get<int>(), d["size"][1].get<int>()};
    Bbox2D box(tl, sz, d.value("timestamp", 0.0f),
               d.value("label", std::string("target")),
               Eigen::Vector2i{img_w, img_h});
    dets.push_back(box);
  }
  return dets;
}

// Synthetic fixtures are rendered as if the camera sits at the robot body
// origin looking straight ahead. The follower reads the pose in the optical
// convention, so that pose is the REP 103 optical -> body quarter turn as [x,
// y, z, w]. An identity here would be read as an optical frame pointing along
// body +z and put every target 90 degrees off.
const Eigen::Vector3f kCameraPosition{0.0f, 0.0f, 0.0f};
const Eigen::Vector4f kCameraRotation{-0.5f, 0.5f, -0.5f, 0.5f};

// The depth PNG as a point cloud in the camera's optical frame: every
// in-range pixel back-projected at its centre, packed as float32 xyz records
// with 4 bytes of padding (point_step 16, offsets 0/4/8) like a LiDAR driver
// publishes. The point-cloud path must then reproduce the depth-image path.
struct PackedCloud {
  std::vector<uint8_t> data;
  int count = 0;

  PointCloudView view() const {
    return PointCloudView{
        ByteSpan(data.data(), data.size()), 16, 16 * count, 1, count, 0, 4, 8};
  }
};

PackedCloud cloud_from_depth(const cv::Mat &depth_mm, const json &cam) {
  const float fx = cam["fx"].get<float>(), fy = cam["fy"].get<float>();
  const float cx = cam["cx"].get<float>(), cy = cam["cy"].get<float>();
  const float to_metres = cam["depth_conversion_factor"].get<float>();
  PackedCloud cloud;
  cloud.data.reserve(static_cast<std::size_t>(depth_mm.total()) * 16);
  for (int row = 0; row < depth_mm.rows; ++row) {
    for (int col = 0; col < depth_mm.cols; ++col) {
      const float d = depth_mm.at<uint16_t>(row, col) * to_metres;
      if (d <= 0.0f) {
        continue;
      }
      const float xyz[3] = {(col + 0.5f - cx) * d / fx,
                            (row + 0.5f - cy) * d / fy, d};
      uint8_t record[16] = {};
      std::memcpy(record, xyz, sizeof(xyz));
      cloud.data.insert(cloud.data.end(), record, record + 16);
      ++cloud.count;
    }
  }
  return cloud;
}

// The cloud is in the optical frame, so its sensor is the camera itself
SensorConfig camera_as_cloud_sensor() {
  SensorConfig sensor;
  sensor.position = kCameraPosition;
  sensor.rotation = kCameraRotation;
  return sensor;
}

std::unique_ptr<Control::RGBDFollower> build_controller(const json &case_json) {
  using namespace Control;
  LinearVelocityControlParams x_params(2.0f, 5.0f, 10.0f);
  LinearVelocityControlParams y_params(1.0f, 3.0f, 5.0f);
  AngularVelocityControlParams angular_params(3.14f, 4.0f, 3.0f, 3.0f);
  ControlLimitsParams ctrl_limits(x_params, y_params, angular_params);

  RGBDFollower::RGBDFollowerConfig config;
  config.setParameter("control_time_step", 0.1);
  config.setParameter("control_horizon", 2);
  config.setParameter("prediction_horizon", 20);
  config.setParameter("use_local_coordinates", true);
  config.setParameter("target_distance", 0.2);
  config.setParameter("target_orientation", 0.0);
  config.setParameter("distance_tolerance", 0.1);
  config.setParameter(
      "depth_conversion_factor",
      case_json["camera"]["depth_conversion_factor"].get<double>());
  config.setParameter("min_depth",
                      case_json["camera"]["min_depth"].get<double>());
  config.setParameter("max_depth",
                      case_json["camera"]["max_depth"].get<double>());

  std::vector<float> robot_dimensions{0.1f, 0.4f};
  auto controller = std::make_unique<RGBDFollower>(
      ControlType::DIFFERENTIAL_DRIVE, ctrl_limits,
      CollisionChecker::ShapeType::CYLINDER, robot_dimensions, kCameraPosition,
      kCameraRotation, config);

  const auto &cam = case_json["camera"];
  controller->setCameraIntrinsics(
      cam["fx"].get<float>(), cam["fy"].get<float>(), cam["cx"].get<float>(),
      cam["cy"].get<float>());
  return controller;
}

// First command of one fixture through one depth source, checked against
// the fixture's expected bounds. Returns (vx, omega) when a command was found.
template <typename DepthSource>
std::optional<std::pair<float, float>>
run_one_source(const FixtureCase &fx, const std::string &source_name,
               const json &case_json, const DepthSource &source,
               Control::RGBDFollower &controller) {
  const auto detections = parse_detections(case_json);
  Path::State state(case_json["robot"]["x"].get<float>(),
                    case_json["robot"]["y"].get<float>(),
                    case_json["robot"]["yaw"].get<float>(),
                    case_json["robot"]["speed"].get<float>());
  controller.setCurrentState(state);
  const std::string tag = fx.name + " (" + source_name + ")";

  const int click_x = case_json["click"]["x"].get<int>();
  const int click_y = case_json["click"]["y"].get<int>();
  const bool init_ok = controller.setInitialTracking(click_x, click_y, source,
                                                     detections, state.yaw);
  const bool expected_init = case_json["expected"]["init_success"].get<bool>();
  BOOST_TEST(init_ok == expected_init, tag << ": setInitialTracking returned "
                                           << init_ok << ", expected "
                                           << expected_init);
  if (!init_ok)
    return std::nullopt;

  Control::Velocity2D current_vel;
  auto result = controller.getTrackingCtrl(source, detections, current_vel);
  BOOST_TEST(result.isTrajFound, tag << ": planner failed to find a control");
  if (!result.isTrajFound)
    return std::nullopt;

  const float vx = result.trajectory.velocities.vx[0];
  const float omega = result.trajectory.velocities.omega[0];

  const auto &exp = case_json["expected"];
  const float vx_min = exp["vx_min"].get<float>();
  const float vx_max = exp["vx_max"].get<float>();
  const float w_min = exp["omega_min"].get<float>();
  const float w_max = exp["omega_max"].get<float>();

  BOOST_TEST(vx >= vx_min, tag << ": vx=" << vx << " < vx_min=" << vx_min);
  BOOST_TEST(vx <= vx_max, tag << ": vx=" << vx << " > vx_max=" << vx_max);
  BOOST_TEST(omega >= w_min,
             tag << ": omega=" << omega << " < omega_min=" << w_min);
  BOOST_TEST(omega <= w_max,
             tag << ": omega=" << omega << " > omega_max=" << w_max);
  return std::make_pair(vx, omega);
}

void run_one_fixture(const FixtureCase &fx) {
  BOOST_TEST_MESSAGE("Running fixture: " << fx.name);
  std::ifstream in((fx.dir / "case.json").string());
  json case_json;
  in >> case_json;

  const cv::Mat depth_mat = load_depth_png(fx.dir / "depth.png");

  auto depth_controller = build_controller(case_json);
  const auto from_depth = run_one_source(
      fx, "depth image", case_json, depth_view(depth_mat), *depth_controller);

  const PackedCloud cloud = cloud_from_depth(depth_mat, case_json["camera"]);
  auto cloud_controller = build_controller(case_json);
  cloud_controller->setPointCloudSensor(camera_as_cloud_sensor());
  const auto from_cloud = run_one_source(fx, "point cloud", case_json,
                                         cloud.view(), *cloud_controller);

  // Same scene through both sources must give the same command
  BOOST_TEST(from_depth.has_value() == from_cloud.has_value(),
             fx.name << ": depth-image and point-cloud paths disagree on "
                        "finding a command");
  if (from_depth && from_cloud) {
    BOOST_CHECK_SMALL(from_depth->first - from_cloud->first, 1e-3f);
    BOOST_CHECK_SMALL(from_depth->second - from_cloud->second, 1e-3f);
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(RGBDFollower_fixture_cases) {
  auto fixtures = discover_fixtures();
  BOOST_REQUIRE_MESSAGE(!fixtures.empty(),
                        "No vision_follower fixtures discovered");
  for (const auto &fx : fixtures) {
    run_one_fixture(fx);
  }
}
