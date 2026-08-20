// CriticalZoneCheckerGPU unit tests.
//
// AdaptiveCpp's runtime is reference-counted: it starts when the first
// SYCL object is constructed and tears down when the last is destroyed
// (AdaptiveCpp/AdaptiveCpp#1233, #1107). Constructing a fresh
// CriticalZoneCheckerGPU per Boost test case restarts the runtime between
// cases, and letting the final instance's destructor run during static
// destruction races the runtime teardown — both surface as glibc heap
// corruption at process exit.
//
// Workaround: one checker per input mode (laserscan, pointcloud) held by
// an intentionally-leaked function-local static. Each test also builds
// its own fresh input state to avoid cross-test dependencies.

#include "test.h"
#include <Eigen/Dense>
#define BOOST_TEST_MODULE KOMPASS CRIT ZONE GPU TESTS
#include "utils/collision_check.h"
#include "utils/critical_zone_check_gpu.h"
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Shared config
// ---------------------------------------------------------------------------

namespace {

constexpr int SCAN_RESOLUTION = 360;
constexpr double SCAN_DEFAULT_RANGE = 10.0;
constexpr float CRIT_ANGLE = 160.0f;
constexpr float CRIT_DIST = 0.3f;
constexpr float SLOW_DIST = 0.6f;
constexpr float LASERSCAN_MIN_H = 0.1f;
constexpr float LASERSCAN_MAX_H = 2.0f;
constexpr float POINTCLOUD_MIN_H = 0.1f;
constexpr float POINTCLOUD_MAX_H = 2.0f;
constexpr float MAX_RANGE = 20.0f;

const auto ROBOT_SHAPE = CollisionChecker::ShapeType::CYLINDER;
const std::vector<float> ROBOT_DIMS{0.51f, 2.0f};

std::vector<double> make_reference_angles() {
  std::vector<double> angles;
  std::vector<double> ranges;
  initLaserscan(SCAN_RESOLUTION, SCAN_DEFAULT_RANGE, ranges, angles);
  return angles;
}

// ---------------------------------------------------------------------------
// Shared, intentionally-leaked checker singletons.
//
// Leaking sidesteps AdaptiveCpp/AdaptiveCpp#1107: running sycl::free and
// sycl::queue destructors during static destruction races the runtime
// teardown and produces glibc heap corruption at exit.
// ---------------------------------------------------------------------------

CriticalZoneCheckerGPU &shared_laserscan_checker() {
  static CriticalZoneCheckerGPU *c = new CriticalZoneCheckerGPU(
      CriticalZoneChecker::InputType::LASERSCAN, ROBOT_SHAPE, ROBOT_DIMS,
      {SensorConfig{Eigen::Vector3f{0.22f, 0.0f, 0.4f},
                    Eigen::Vector4f{0.0f, 0.0f, 0.99f, 0.0f}}},
      CRIT_ANGLE, CRIT_DIST, SLOW_DIST, LASERSCAN_MIN_H, LASERSCAN_MAX_H,
      MAX_RANGE, make_reference_angles());
  return *c;
}

CriticalZoneCheckerGPU &shared_pointcloud_checker() {
  static CriticalZoneCheckerGPU *c = new CriticalZoneCheckerGPU(
      CriticalZoneChecker::InputType::POINTCLOUD, ROBOT_SHAPE, ROBOT_DIMS,
      {SensorConfig{}}, CRIT_ANGLE, CRIT_DIST, SLOW_DIST, POINTCLOUD_MIN_H,
      POINTCLOUD_MAX_H, MAX_RANGE);
  return *c;
}

// Front + back mounted sensors (fusion tests). Mount heights 0.2 m; the
// body-frame band [POINTCLOUD_MIN_H, POINTCLOUD_MAX_H] is shared
CriticalZoneCheckerGPU &shared_multi_checker() {
  static CriticalZoneCheckerGPU *c = new CriticalZoneCheckerGPU(
      CriticalZoneChecker::InputType::POINTCLOUD, ROBOT_SHAPE, ROBOT_DIMS,
      {SensorConfig::fromYaw({0.2f, 0.0f, 0.2f}, 0.0f),
       SensorConfig::fromYaw({-0.2f, 0.0f, 0.2f}, static_cast<float>(M_PI))},
      CRIT_ANGLE, CRIT_DIST, SLOW_DIST, POINTCLOUD_MIN_H, POINTCLOUD_MAX_H,
      MAX_RANGE);
  return *c;
}

// ---------------------------------------------------------------------------
// Helpers: build a fresh laserscan input for each test to avoid cross-test
// state leakage.
// ---------------------------------------------------------------------------

struct LaserScanInput {
  std::vector<double> ranges;
  std::vector<double> angles;
};

LaserScanInput fresh_laserscan() {
  LaserScanInput s;
  initLaserscan(SCAN_RESOLUTION, SCAN_DEFAULT_RANGE, s.ranges, s.angles);
  return s;
}

// Point-cloud byte-offset constants (PointXYZ is in test.h).
const int PC_POINT_STEP = sizeof(PointXYZ);
const int PC_X_OFF = offsetof(PointXYZ, x);
const int PC_Y_OFF = offsetof(PointXYZ, y);
const int PC_Z_OFF = offsetof(PointXYZ, z);

float run_pc_check(const std::vector<uint8_t> &cloud, bool forward) {
  const int num_points = static_cast<int>(cloud.size() / PC_POINT_STEP);
  const int width = num_points;
  const int height = num_points > 0 ? 1 : 0;
  const int row_step = width * PC_POINT_STEP;
  return shared_pointcloud_checker().check(cloud, PC_POINT_STEP, row_step,
                                           height, width, PC_X_OFF, PC_Y_OFF,
                                           PC_Z_OFF, forward);
}

} // namespace

// ===========================================================================
// LASERSCAN tests (1-8)
// ===========================================================================

BOOST_AUTO_TEST_CASE(test_laserscan_behind_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(0.0, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(0.1, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(-0.1, 0.2, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ true);
  BOOST_TEST(result == 1.0,
             "Angles behind, moving forward -> expected 1.0, got " << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_front_far_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ true);
  BOOST_TEST(result == 1.0,
             "Angles in front, far, moving forward -> expected 1.0, got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_front_close_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(M_PI, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(M_PI + 0.1, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(M_PI - 0.1, 0.2, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ true);
  BOOST_TEST(result == 0.0,
             "Angles in front, close, moving forward -> expected 0.0, got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_front_close_moving_backward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(M_PI, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(M_PI + 0.1, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(M_PI - 0.1, 0.2, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ false);
  BOOST_TEST(result == 1.0,
             "Angles in front, close, moving backward -> expected 1.0, got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_back_close_moving_backward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(0.0, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(0.1, 0.2, scan.ranges, scan.angles);
  setLaserscanAtAngle(-0.1, 0.2, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ false);
  BOOST_TEST(result == 0.0,
             "Angles behind, close, moving backward -> expected 0.0, got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_back_slowdown_moving_backward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(0.0, 1.3, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ false);
  BOOST_TEST((result > 0.0 && result < 1.0),
             "Angle behind in slowdown zone, moving backward -> expected in "
             "(0, 1), got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_back_slowdown_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(0.0, 1.3, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ true);
  BOOST_TEST(result == 1.0,
             "Angle behind in slowdown zone, moving forward -> expected 1.0, "
             "got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_front_slowdown_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(M_PI, 0.7, scan.ranges, scan.angles);

  float result =
      shared_laserscan_checker().check(toVecF(scan.ranges), /*forward*/ true);
  BOOST_TEST((result > 0.0 && result < 1.0),
             "Angle in front in slowdown zone, moving forward -> expected in "
             "(0, 1), got "
                 << result);
}

// ===========================================================================
// POINTCLOUD tests (9-14)
// ===========================================================================

BOOST_AUTO_TEST_CASE(test_pointcloud_empty_is_safe) {
  Timer time;
  std::vector<uint8_t> cloud;
  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 1.0f, "Empty point cloud should be safe (1.0)");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_critical_obstacle_front) {
  Timer time;
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.7f, 0.0f, 0.5f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 0.0f, "Point at 0.7 m should trigger stop (0.0)");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_height_filter_drops_point) {
  Timer time;
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.7f, 0.0f, /*z above max*/ 3.0f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 1.0f, "High point (> max_z) should be ignored");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_slowdown_zone) {
  Timer time;
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.95f, 0.0f, 0.5f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST((result > 0.4f && result < 0.6f),
             "Point in slowdown zone -> expected ~0.5, got " << result);
}

BOOST_AUTO_TEST_CASE(test_pointcloud_mixed_stop_wins) {
  Timer time;
  std::vector<uint8_t> cloud;
  // Slowdown candidates
  addPointToCloud(cloud, 0.95f, 0.0f, 0.5f);
  addPointToCloud(cloud, 1.0f, 1.0f, 0.5f);
  addPointToCloud(cloud, -1.0f, -1.0f, 0.5f);
  // Filtered by Z (height out of range)
  addPointToCloud(cloud, -0.1f, -0.1f, 3.0f);
  addPointToCloud(cloud, -0.1f, -0.1f, -3.0f);
  addPointToCloud(cloud, 0.1f, 0.2f, 4.0f);
  addPointToCloud(cloud, 0.1f, 0.2f, -4.0f);
  // Stop-zone point — should dominate the min-reduction
  addPointToCloud(cloud, 0.75f, 0.0f, 0.5f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 0.0f,
             "Stop-zone point should win min reduction, got " << result);
}

BOOST_AUTO_TEST_CASE(test_pointcloud_mixed_slowdown_backward) {
  Timer time;
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.95f, 0.0f, 0.5f);
  addPointToCloud(cloud, -0.95f, 0.0f, 0.5f);
  addPointToCloud(cloud, 1.0f, 1.0f, 0.5f);
  addPointToCloud(cloud, -1.0f, -1.0f, 0.5f);
  // Filtered by Z
  addPointToCloud(cloud, -0.1f, -0.1f, 3.0f);
  addPointToCloud(cloud, -0.1f, -0.1f, -3.0f);
  addPointToCloud(cloud, 0.1f, 0.2f, 4.0f);
  addPointToCloud(cloud, 0.1f, 0.2f, -4.0f);

  float result = run_pc_check(cloud, /*forward*/ false);
  BOOST_TEST((result > 0.4f && result < 0.6f),
             "Mixed cloud, moving backward -> expected slowdown ~0.5, got "
                 << result);
}

// ===========================================================================
// MULTI-SENSOR POINTCLOUD tests (15-21): front + back mounts, see
// shared_multi_checker above
// ===========================================================================

PointCloudView make_view(const std::vector<uint8_t> &cloud) {
  const int n = static_cast<int>(cloud.size() / PC_POINT_STEP);
  return PointCloudView{cloud, PC_POINT_STEP, n * PC_POINT_STEP, n > 0 ? 1 : 0,
                        n,     PC_X_OFF,      PC_Y_OFF,          PC_Z_OFF};
}

// Back-Sensor Obstacle & Moving Backward (Headline Case) ---
// An obstacle seen ONLY by the back sensor must stop reverse motion and
// leave forward motion untouched. Back-sensor frame (0.55, 0, 0.3) ->
// body (-0.75, 0, 0.5). Dist = 0.75. 0.75 - Radius(0.51) = 0.24 <
// Crit(0.3) -> stop when backing up.
BOOST_AUTO_TEST_CASE(test_multi_sensor_back_obstacle_stops_reverse) {
  Timer time;
  std::vector<uint8_t> back_cloud;
  addPointToCloud(back_cloud, 0.55f, 0.0f, 0.3f);
  const std::vector<uint8_t> empty_front;

  auto &checker = shared_multi_checker();
  const float backward =
      checker.check({make_view(empty_front), make_view(back_cloud)},
                    /*forward*/ false);
  BOOST_TEST(backward == 0.0f,
             "back obstacle must stop reverse motion, got " << backward);

  const float forward = checker.check(
      {make_view(empty_front), make_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST(forward == 1.0f,
             "back obstacle must not affect forward motion, got " << forward);
}

// --- Critical Cone Is a Body-Frame Gate ---
// The cone is not a per-sensor gate: a point in the BACK sensor's cloud
// that lands ahead of the robot must trip the forward check. Back-sensor
// frame (-0.9, 0, 0.3) -> body (0.7, 0, 0.5). 0.7 - Radius(0.51) = 0.19 <
// Crit(0.3) -> forward stop from the back sensor's data.
BOOST_AUTO_TEST_CASE(test_multi_sensor_cone_is_body_frame) {
  Timer time;
  std::vector<uint8_t> back_cloud;
  addPointToCloud(back_cloud, -0.9f, 0.0f, 0.3f);
  const std::vector<uint8_t> empty_front;

  const float forward = shared_multi_checker().check(
      {make_view(empty_front), make_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST(forward == 0.0f,
             "a back-sensor point ahead of the robot must trip the forward "
             "check, got "
                 << forward);
}

// --- Min-Wins Fusion Across Clouds ---
// Front cloud holds a slowdown-zone point: front-sensor (0.75, 0, 0.3) ->
// body (0.95, 0, 0.5), factor (0.95 - 0.51 - 0.3) / 0.3 = ~0.467. The
// back cloud is far. Fused factor equals the slowdown factor; adding a
// closer front point lowers it to 0.
BOOST_AUTO_TEST_CASE(test_multi_sensor_min_across_clouds) {
  Timer time;
  std::vector<uint8_t> front_cloud;
  addPointToCloud(front_cloud, 0.75f, 0.0f, 0.3f);
  std::vector<uint8_t> back_cloud;
  addPointToCloud(back_cloud, 5.0f, 0.0f, 0.3f); // far, safe

  auto &checker = shared_multi_checker();
  const float slow = checker.check(
      {make_view(front_cloud), make_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST((slow > 0.4f && slow < 0.55f),
             "expected slowdown factor ~0.467, got " << slow);

  addPointToCloud(front_cloud, 0.6f, 0.0f, 0.3f); // body 0.8 -> critical
  const float stop = checker.check(
      {make_view(front_cloud), make_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST(stop == 0.0f, "critical point must win the min, got " << stop);
}

// --- Repeat-Call State ---
// A stop result must not leak into the next call (guards the shared-result
// reset being hoisted before the batch's submits).
BOOST_AUTO_TEST_CASE(test_multi_sensor_repeat_call_state) {
  Timer time;
  std::vector<uint8_t> danger;
  addPointToCloud(danger, 0.55f, 0.0f, 0.3f); // front: body 0.75 -> stop
  std::vector<uint8_t> safe;
  addPointToCloud(safe, 5.0f, 0.0f, 0.3f);

  auto &checker = shared_multi_checker();
  const float stop =
      checker.check({make_view(danger), make_view(safe)}, /*forward*/ true);
  BOOST_TEST(stop == 0.0f);

  const float clear =
      checker.check({make_view(safe), make_view(safe)}, /*forward*/ true);
  BOOST_TEST(clear == 1.0f,
             "stop from the previous call leaked into this one, got " << clear);

  // All-empty batch -> nothing observed -> no constraint
  const std::vector<uint8_t> empty;
  const float idle =
      checker.check({make_view(empty), make_view(empty)}, /*forward*/ true);
  BOOST_TEST(idle == 1.0f);
}

// --- Tilted Mount, Exact Values ---
// 25 deg pitch at z=0.3 seeing sensor-frame (0.8, 0, 0.3). With the full
// transform the point's z leans INTO x:
// x_body = 0.8*cos25 + 0.3*sin25 = 0.852 -> slowdown (factor ~0.14).
// The old kernel dropped the z column (x_body = 0.725 -> false STOP), so
// a 0 here means the tilt terms regressed.
BOOST_AUTO_TEST_CASE(test_multi_sensor_tilted_mount_exact) {
  Timer time;
  const float half_pitch = 12.5f * static_cast<float>(M_PI) / 180.0f;
  SensorConfig tilted;
  tilted.position = {0.0f, 0.0f, 0.3f};
  tilted.rotation = {0.0f, std::sin(half_pitch), 0.0f, std::cos(half_pitch)};

  static CriticalZoneCheckerGPU *tilted_checker = new CriticalZoneCheckerGPU(
      CriticalZoneChecker::InputType::POINTCLOUD, ROBOT_SHAPE, ROBOT_DIMS,
      {tilted}, CRIT_ANGLE, CRIT_DIST, SLOW_DIST, POINTCLOUD_MIN_H,
      POINTCLOUD_MAX_H, MAX_RANGE);

  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.8f, 0.0f, 0.3f);
  const float factor =
      tilted_checker->check({make_view(cloud)}, /*forward*/ true);
  BOOST_TEST((factor > 0.05f && factor < 0.25f),
             "expected slowdown ~0.14 under the full tilt transform, got "
                 << factor);
}

// --- N=1 Adapter ---
// The  single-cloud byte entry is an N=1 adapter onto the batched
// path: identical result for the identical cloud.
BOOST_AUTO_TEST_CASE(test_multi_sensor_n1_adapter) {
  Timer time;
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.95f, 0.0f, 0.5f);
  addPointToCloud(cloud, -1.0f, -1.0f, 0.5f);
  addPointToCloud(cloud, 0.75f, 0.0f, 0.5f);

  const float legacy = run_pc_check(cloud, /*forward*/ true);
  const float batched =
      shared_pointcloud_checker().check({make_view(cloud)}, /*forward*/ true);
  BOOST_TEST(legacy == batched, "adapter and batched entry disagree: "
                                    << legacy << " vs " << batched);
}

// --- Error Paths ---
// Cloud-count mismatch, negative offsets named per cloud, and mode guards
// on both entries.
BOOST_AUTO_TEST_CASE(test_multi_sensor_error_paths) {
  Timer time;
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 5.0f, 0.0f, 0.3f);

  auto &checker = shared_multi_checker();

  BOOST_CHECK_THROW(checker.check({make_view(cloud)}, true),
                    std::invalid_argument);

  PointCloudView bad = make_view(cloud);
  bad.y_offset = -4;
  BOOST_CHECK_EXCEPTION(
      checker.check({make_view(cloud), bad}, true), std::invalid_argument,
      [](const std::invalid_argument &e) {
        return std::string(e.what()).find("clouds[1]") != std::string::npos &&
               std::string(e.what()).find("non-negative") != std::string::npos;
      });

  // Laserscan entry on a pointcloud-mode checker throws (was a silent
  // memcpy into a null device pointer)
  Eigen::VectorXf ranges(4);
  ranges.setConstant(1.0f);
  BOOST_CHECK_THROW(checker.check(ranges, true), std::logic_error);

  // Batched entry on a laserscan-mode checker throws
  BOOST_CHECK_THROW(shared_laserscan_checker().check({make_view(cloud)}, true),
                    std::logic_error);
}

// --- Test 22: Optical-Axis Point Is Not Filtered (Depth Camera) ---
// Depth camera in the ROS optical frame (z forward, x right, y down),
// mounted at (0.1, 0, 0.3): a point ON the optical axis is straight ahead
// of the robot and must trip the forward check. Sensor (0, 0, 0.4) ->
// body (0.5, 0, 0.3): 0.5 - Radius(0.51) = -0.01 < Crit(0.3) -> Critical.
// A TRUE origin point must still be rejected (it maps onto the mount position
// inside the robot).
BOOST_AUTO_TEST_CASE(test_multi_sensor_optical_axis_not_filtered) {
  Timer time;
  SensorConfig camera;
  camera.position = {0.1f, 0.0f, 0.3f};
  // Body-from-optical: x_body = z_opt, y_body = -x_opt, z_body = -y_opt
  camera.rotation = {-0.5f, 0.5f, -0.5f, 0.5f};

  static CriticalZoneCheckerGPU *camera_checker = new CriticalZoneCheckerGPU(
      CriticalZoneChecker::InputType::POINTCLOUD, ROBOT_SHAPE, ROBOT_DIMS,
      {camera}, CRIT_ANGLE, CRIT_DIST, SLOW_DIST, POINTCLOUD_MIN_H,
      POINTCLOUD_MAX_H, MAX_RANGE);

  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.0f, 0.0f, 0.4f);
  const float factor = camera_checker->check({make_view(cloud)},
                                             /*forward*/ true);
  BOOST_TEST(factor == 0.0f,
             "On-axis point 40 cm ahead of a depth camera -> expected 0.0, "
             "got "
                 << factor);

  std::vector<uint8_t> origin_cloud;
  addPointToCloud(origin_cloud, 0.0f, 0.0f, 0.0f);
  const float clear = camera_checker->check({make_view(origin_cloud)},
                                            /*forward*/ true);
  BOOST_TEST(clear == 1.0f,
             "Sensor-origin self-return should be filtered -> expected 1.0, "
             "got "
                 << clear);
}
