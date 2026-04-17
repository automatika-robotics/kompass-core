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
// its own fresh input state to avoid cross-test dependencies. Same
// pattern in mapper_test_gpu.cpp and pointcloud_to_laserscan_test_gpu.cpp.

#include "test.h"
#include <Eigen/Dense>
#define BOOST_TEST_MODULE KOMPASS CRIT ZONE GPU TESTS
#include "utils/collision_check.h"
#include "utils/critical_zone_check_gpu.h"
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Shared config (same values as the original single-case test).
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
      /*sensor_pos*/ Eigen::Vector3f{0.22f, 0.0f, 0.4f},
      /*sensor_rot*/ Eigen::Vector4f{0.0f, 0.0f, 0.99f, 0.0f},
      CRIT_ANGLE, CRIT_DIST, SLOW_DIST, make_reference_angles(),
      LASERSCAN_MIN_H, LASERSCAN_MAX_H, MAX_RANGE);
  return *c;
}

CriticalZoneCheckerGPU &shared_pointcloud_checker() {
  static CriticalZoneCheckerGPU *c = new CriticalZoneCheckerGPU(
      CriticalZoneChecker::InputType::POINTCLOUD, ROBOT_SHAPE, ROBOT_DIMS,
      /*sensor_pos*/ Eigen::Vector3f{0.0f, 0.0f, 0.0f},
      /*sensor_rot*/ Eigen::Vector4f{0.0f, 0.0f, 0.0f, 1.0f},
      CRIT_ANGLE, CRIT_DIST, SLOW_DIST, make_reference_angles(),
      POINTCLOUD_MIN_H, POINTCLOUD_MAX_H, MAX_RANGE);
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

float run_pc_check(const std::vector<int8_t> &cloud, bool forward) {
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

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ true);
  BOOST_TEST(result == 1.0,
             "Angles behind, moving forward -> expected 1.0, got " << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_front_far_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ true);
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

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ true);
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

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ false);
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

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ false);
  BOOST_TEST(result == 0.0,
             "Angles behind, close, moving backward -> expected 0.0, got "
                 << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_back_slowdown_moving_backward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(0.0, 1.3, scan.ranges, scan.angles);

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ false);
  BOOST_TEST((result > 0.0 && result < 1.0),
             "Angle behind in slowdown zone, moving backward -> expected in "
             "(0, 1), got " << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_back_slowdown_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(0.0, 1.3, scan.ranges, scan.angles);

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ true);
  BOOST_TEST(result == 1.0,
             "Angle behind in slowdown zone, moving forward -> expected 1.0, "
             "got " << result);
}

BOOST_AUTO_TEST_CASE(test_laserscan_front_slowdown_moving_forward) {
  Timer time;
  auto scan = fresh_laserscan();
  setLaserscanAtAngle(M_PI, 0.7, scan.ranges, scan.angles);

  float result = shared_laserscan_checker().check(scan.ranges, /*forward*/ true);
  BOOST_TEST((result > 0.0 && result < 1.0),
             "Angle in front in slowdown zone, moving forward -> expected in "
             "(0, 1), got " << result);
}

// ===========================================================================
// POINTCLOUD tests (9-14)
// ===========================================================================

BOOST_AUTO_TEST_CASE(test_pointcloud_empty_is_safe) {
  Timer time;
  std::vector<int8_t> cloud;
  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 1.0f, "Empty point cloud should be safe (1.0)");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_critical_obstacle_front) {
  Timer time;
  std::vector<int8_t> cloud;
  addPointToCloud(cloud, 0.7f, 0.0f, 0.5f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 0.0f, "Point at 0.7 m should trigger stop (0.0)");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_height_filter_drops_point) {
  Timer time;
  std::vector<int8_t> cloud;
  addPointToCloud(cloud, 0.7f, 0.0f, /*z above max*/ 3.0f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST(result == 1.0f, "High point (> max_z) should be ignored");
}

BOOST_AUTO_TEST_CASE(test_pointcloud_slowdown_zone) {
  Timer time;
  std::vector<int8_t> cloud;
  addPointToCloud(cloud, 0.95f, 0.0f, 0.5f);

  float result = run_pc_check(cloud, /*forward*/ true);
  BOOST_TEST((result > 0.4f && result < 0.6f),
             "Point in slowdown zone -> expected ~0.5, got " << result);
}

BOOST_AUTO_TEST_CASE(test_pointcloud_mixed_stop_wins) {
  Timer time;
  std::vector<int8_t> cloud;
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
  BOOST_TEST(result == 0.0f, "Stop-zone point should win min reduction, got "
                                 << result);
}

BOOST_AUTO_TEST_CASE(test_pointcloud_mixed_slowdown_backward) {
  Timer time;
  std::vector<int8_t> cloud;
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
