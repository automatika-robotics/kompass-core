// Parametrized fixture-based test for VisionDWA.
//
// Loads each fixture under tests/resources/vision_dwa/<case>/ which contains:
//   - depth.png: 16-bit single-channel depth image (millimeters)
//   - case.json: camera intrinsics, robot state, 2D detections, click pixel,
//                and loose expected bounds for the resulting control command.
//
// Mirrors tests/test_vision_dwa_follower.py so both layers exercise the same
// data. To add cases, edit tests/resources/vision_dwa/generate_fixtures.py and
// re-run it (or drop a new fixture directory in by hand).

#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/tracking.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"

#define BOOST_TEST_MODULE VISION_DWA_FIXTURE_TESTS
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>
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
// tests/resources/vision_dwa.
fs::path locate_fixture_root() {
  fs::path here = boost::dll::program_location().parent_path();
  for (int i = 0; i < 8; ++i) {
    fs::path candidate = here / "tests" / "resources" / "vision_dwa";
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
  fs::path src_default = fs::path(__FILE__).parent_path().parent_path()
                             .parent_path().parent_path() /
                         "tests" / "resources" / "vision_dwa";
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
    if (!fs::is_directory(entry.path())) continue;
    if (!fs::exists(entry.path() / "case.json")) continue;
    if (!fs::exists(entry.path() / "depth.png")) continue;
    out.push_back({entry.path().filename().string(), entry.path()});
  }
  std::sort(out.begin(), out.end(),
            [](const FixtureCase &a, const FixtureCase &b) {
              return a.name < b.name;
            });
  return out;
}

Eigen::MatrixX<unsigned short> load_depth_png(const fs::path &png_path) {
  cv::Mat raw = cv::imread(png_path.string(), cv::IMREAD_UNCHANGED);
  BOOST_REQUIRE_MESSAGE(!raw.empty(),
                        "Could not load depth.png at " << png_path.string());
  BOOST_REQUIRE_MESSAGE(raw.type() == CV_16UC1,
                        "depth.png must be 16-bit single-channel: "
                            << png_path.string());
  Eigen::MatrixX<unsigned short> depth(raw.rows, raw.cols);
  for (int r = 0; r < raw.rows; ++r) {
    for (int c = 0; c < raw.cols; ++c) {
      depth(r, c) = raw.at<unsigned short>(r, c);
    }
  }
  return depth;
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

std::unique_ptr<Control::VisionDWA> build_controller(const json &case_json) {
  using namespace Control;
  LinearVelocityControlParams x_params(1.5f, 3.0f, 3.0f);
  LinearVelocityControlParams y_params(1.0f, 3.0f, 3.0f);
  AngularVelocityControlParams angular_params(M_PI / 2, 2.5f, 2.5f, 2.5f);
  ControlLimitsParams ctrl_limits(x_params, y_params, angular_params);

  CostEvaluator::TrajectoryCostsWeights cost_weights;
  cost_weights.setParameter("reference_path_distance_weight", 1.0);
  cost_weights.setParameter("goal_distance_weight", 1.0);
  cost_weights.setParameter("obstacles_distance_weight", 0.0);
  cost_weights.setParameter("smoothness_weight", 0.0);
  cost_weights.setParameter("jerk_weight", 0.0);

  VisionDWA::VisionDWAConfig config;
  config.setParameter("control_time_step", 0.1);
  config.setParameter("control_horizon", 2);
  config.setParameter("prediction_horizon", 6);
  config.setParameter("use_local_coordinates", true);
  config.setParameter("target_distance", 0.5);
  config.setParameter("distance_tolerance", 0.1);
  config.setParameter(
      "depth_conversion_factor",
      case_json["camera"]["depth_conversion_factor"].get<double>());
  config.setParameter("min_depth",
                      case_json["camera"]["min_depth"].get<double>());
  config.setParameter("max_depth",
                      case_json["camera"]["max_depth"].get<double>());

  std::vector<float> robot_dimensions{0.1f, 0.4f};
  Eigen::Vector3f prox_pos{0.0f, 0.0f, 0.0f};
  Eigen::Vector4f prox_rot{0.0f, 0.0f, 0.0f, 1.0f};
  Eigen::Vector3f cam_pos{0.0f, 0.0f, 0.0f};
  Eigen::Vector4f cam_rot{0.0f, 0.0f, 0.0f, 1.0f};

  auto controller = std::make_unique<VisionDWA>(
      ControlType::DIFFERENTIAL_DRIVE, ctrl_limits, 15, 15,
      CollisionChecker::ShapeType::CYLINDER, robot_dimensions, prox_pos,
      prox_rot, cam_pos, cam_rot, 0.1, cost_weights, 1, config);

  const auto &cam = case_json["camera"];
  controller->setCameraIntrinsics(cam["fx"].get<float>(),
                                  cam["fy"].get<float>(),
                                  cam["cx"].get<float>(),
                                  cam["cy"].get<float>());
  return controller;
}

void run_one_fixture(const FixtureCase &fx) {
  BOOST_TEST_MESSAGE("Running fixture: " << fx.name);
  std::ifstream in((fx.dir / "case.json").string());
  json case_json;
  in >> case_json;

  auto depth = load_depth_png(fx.dir / "depth.png");
  auto detections = parse_detections(case_json);
  auto controller = build_controller(case_json);

  Path::State state(case_json["robot"]["x"].get<float>(),
                    case_json["robot"]["y"].get<float>(),
                    case_json["robot"]["yaw"].get<float>(),
                    case_json["robot"]["speed"].get<float>());
  controller->setCurrentState(state);

  const int click_x = case_json["click"]["x"].get<int>();
  const int click_y = case_json["click"]["y"].get<int>();
  const bool init_ok = controller->setInitialTracking(click_x, click_y, depth,
                                                      detections, state.yaw);
  const bool expected_init = case_json["expected"]["init_success"].get<bool>();
  BOOST_TEST(init_ok == expected_init,
             fx.name << ": setInitialTracking returned " << init_ok
                     << ", expected " << expected_init);
  if (!init_ok) return;

  Control::Velocity2D current_vel;
  std::vector<Eigen::Vector3f> empty_cloud;
  auto result = controller->getTrackingCtrl(depth, detections, current_vel,
                                            empty_cloud);
  BOOST_TEST(result.isTrajFound,
             fx.name << ": planner failed to find a control");
  if (!result.isTrajFound) return;

  const float vx = result.trajectory.velocities.vx[0];
  const float omega = result.trajectory.velocities.omega[0];

  const auto &exp = case_json["expected"];
  const float vx_min = exp["vx_min"].get<float>();
  const float vx_max = exp["vx_max"].get<float>();
  const float w_min = exp["omega_min"].get<float>();
  const float w_max = exp["omega_max"].get<float>();

  BOOST_TEST(vx >= vx_min,
             fx.name << ": vx=" << vx << " < vx_min=" << vx_min);
  BOOST_TEST(vx <= vx_max,
             fx.name << ": vx=" << vx << " > vx_max=" << vx_max);
  BOOST_TEST(omega >= w_min,
             fx.name << ": omega=" << omega << " < omega_min=" << w_min);
  BOOST_TEST(omega <= w_max,
             fx.name << ": omega=" << omega << " > omega_max=" << w_max);
}

}  // namespace

BOOST_AUTO_TEST_CASE(VisionDWA_fixture_cases) {
  auto fixtures = discover_fixtures();
  BOOST_REQUIRE_MESSAGE(!fixtures.empty(),
                        "No vision_dwa fixtures discovered");
  for (const auto &fx : fixtures) {
    run_one_fixture(fx);
  }
}
