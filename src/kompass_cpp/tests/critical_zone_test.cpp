#define BOOST_TEST_MODULE KOMPASS_CRITICAL_ZONE_TESTS
#include "test.h"
#include "utils/critical_zone_check.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstring> // For memcpy
#include <vector>

using namespace Kompass;

BOOST_AUTO_TEST_CASE(test_critical_zone_check) {
  // Shared Setup
  auto robotShapeType = CollisionChecker::ShapeType::CYLINDER;
  CriticalZoneChecker::InputType inputType =
      CriticalZoneChecker::InputType::LASERSCAN;
  std::vector<float> robotDimensions{0.51, 2.0};

  const Eigen::Vector3f sensor_position_body{0.22, 0.0, 0.4};
  const Eigen::Vector4f sensor_rotation_body{0, 0, 0.99, 0.0};

  // Robot laserscan value holders
  std::vector<double> scan_angles;
  std::vector<double> scan_ranges;
  initLaserscan(360, 10.0, scan_ranges, scan_angles);

  float critical_angle = 160.0, critical_distance = 0.3,
        slowdown_distance = 0.6;
  float min_height = 0.1, max_height = 2.0;

  CriticalZoneChecker zoneChecker(
      inputType, robotShapeType, robotDimensions,
      {SensorConfig{sensor_position_body, sensor_rotation_body}},
      critical_angle, critical_distance, slowdown_distance, min_height,
      max_height, 20.0, scan_angles);

  LOG_INFO("Testing Emergency Stop with CPU (LASERSCAN)");

  // --- Test 1: Behind & Moving Forward ---
  {
    Timer time;
    bool forward_motion = true;
    setLaserscanAtAngle(0.0, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(0.1, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(-0.1, 0.2, scan_ranges, scan_angles);

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(result == 1.0,
               "Angles are behind and robot is moving forward -> "
               "Critical zone result should be 1.0, returned "
                   << result);
    if (result == 1.0) {
      LOG_INFO("Test1 PASSED: Angles are behind and robot is moving forward");
    }
  }

  // --- Test 2: Front Far & Moving Forward ---
  {
    Timer time;
    bool forward_motion = true;
    initLaserscan(360, 10.0, scan_ranges, scan_angles); // Reset scan

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(result == 1.0,
               "Angles are in front and far and robot is moving forward "
               "-> Critical zone result should be 1.0, returned "
                   << result);
    if (result == 1.0) {
      LOG_INFO("Test2 PASSED: Angles are in front and robot is moving forward");
    }
  }

  // --- Test 3: Front Close & Moving Forward ---
  {
    Timer time;
    bool forward_motion = true;
    setLaserscanAtAngle(M_PI, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(M_PI + 0.1, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(M_PI - 0.1, 0.2, scan_ranges, scan_angles);

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(result == 0.0,
               "Angles are in front and close and robot is moving "
               "forward -> Critical zone result should be 0.0, returned "
                   << result);
    if (result == 0.0) {
      LOG_INFO(
          "Test3 PASSED: Angles are in front and close and robot is moving "
          "forward");
    }
  }

  // --- Test 4: Front Close & Moving Backward ---
  {
    Timer time;
    bool forward_motion = false;
    // Note: Ranges are still set from Test 3

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(result == 1.0,
               "Angles are in front and close and robot is moving "
               "backwards-> Critical zone result should be 1.0, returned "
                   << result);
    if (result == 1.0) {
      LOG_INFO("Test4 PASSED: Angles are in front and close and robot is "
               "moving backward");
    }
  }

  // --- Test 5: Back Close & Moving Backward ---
  {
    Timer time;
    bool forward_motion = false;
    setLaserscanAtAngle(0.0, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(0.1, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(-0.1, 0.2, scan_ranges, scan_angles);

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(result == 0.0,
               "Angles are in back and close and robot is moving "
               "backwards -> Critical zone result should be 0.0, returned "
                   << result);
    if (result == 0.0) {
      LOG_INFO("Test5 PASSED: Angles are in back and close and robot is moving "
               "backwards");
    }
  }

  // --- Test 6: Back Slowdown & Moving Backward ---
  {
    Timer time;
    bool forward_motion = false;
    initLaserscan(360, 10.0, scan_ranges, scan_angles);
    setLaserscanAtAngle(0.0, 1.3, scan_ranges, scan_angles);

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(
        (result > 0.0 and result < 1.0),
        "Angles are in back and in the slowdown zone and robot is moving "
        "backwards -> Critical zone result should be between [0, 1], returned "
            << result);
    if (result > 0.0 and result < 1.0) {
      LOG_INFO("Test6 PASSED: Angles are in back and in the slowdown zone and "
               "robot is moving "
               "backwards, slowdown factor = ",
               result);
    }
  }

  // --- Test 7: Back Slowdown & Moving Forward ---
  {
    Timer time;
    bool forward_motion = true;
    // Ranges are still set from Test 6

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(
        result == 1.0,
        "Angles are in back and in the slowdown zone and robot is moving "
        "forward -> Critical zone result should be between 1.0, returned "
            << result);
    if (result == 1.0) {
      LOG_INFO("Test7 PASSED: Angles are in back and in the slowdown zone and "
               "robot is moving "
               "forward, slowdown factor = ",
               result);
    }
  }

  // --- Test 8: Front Slowdown & Moving Forward ---
  {
    Timer time;
    bool forward_motion = true;
    setLaserscanAtAngle(M_PI, 0.7, scan_ranges, scan_angles);

    float result = zoneChecker.check(toVecF(scan_ranges), forward_motion);
    BOOST_TEST(
        (result > 0.0 and result < 1.0),
        "Angles are in front and in the slowdown zone and robot is moving "
        "forward -> Critical zone result should be between [0, 1], returned "
            << result);
    if (result > 0.0 and result < 1.0) {
      LOG_INFO("Test8 PASSED: Angles are in front and in the slowdown zone and "
               "robot is moving "
               "forward, slowdown factor = ",
               result);
    }
  }

  // ==========================================
  //      POINT CLOUD TESTS
  // ==========================================

  LOG_INFO("Testing Emergency Stop with CPU (POINTCLOUD)");

  // Instantiate a separate checker for PointCloud (identity mount; the
  // body-frame band coincides with the sensor frame here)
  CriticalZoneChecker pcChecker(CriticalZoneChecker::InputType::POINTCLOUD,
                                robotShapeType, robotDimensions,
                                {SensorConfig{}}, critical_angle,
                                critical_distance /*crit_dist*/,
                                slowdown_distance /*slow_dist*/,
                                0.1 /*min_h*/, 2.0 /*max_h*/, 20.0);

  std::vector<uint8_t> cloud_data;

  // Use sizeof and offsetof to guarantee alignment matches the helper
  int point_step = sizeof(PointXYZ);
  int x_off = offsetof(PointXYZ, x);
  int y_off = offsetof(PointXYZ, y);
  int z_off = offsetof(PointXYZ, z);

  // Helper lambda to run check with dynamic width calculation
  auto run_pc_check = [&](bool forward) -> float {
    int num_points = cloud_data.size() / point_step;
    int width = num_points;
    int height = 1;
    int row_step = width * point_step; // Dynamic row_step calculation

    return pcChecker.check(cloud_data, point_step, row_step, height, width,
                           x_off, y_off, z_off, forward);
  };

  // --- Test 9: Empty Cloud (Safe) ---
  {
    Timer time;
    cloud_data.clear();
    // width will be 0, safe
    float result = run_pc_check(true);
    BOOST_TEST(result == 1.0, "Empty PointCloud should be safe (1.0)");
  }

  // --- Test 10: Critical Obstacle (Front) ---
  {
    Timer time;
    cloud_data.clear();
    // Front: x=0.8, y=0, z=0.5.
    // Dist = 0.8. RobotRad(0.5) + Crit(0.3) = 0.8 -> 0.7 < 0.8 -> Critical.
    addPointToCloud(cloud_data, 0.7f, 0.0f, 0.5f);

    float result = run_pc_check(true);
    BOOST_TEST(result == 0.0, "Point at 0.7m, should trigger stop (0.0)");

    if (result == 0.0)
      LOG_INFO("Test10 PASSED: PointCloud Critical Stop");
  }

  // --- Test 11: Height Filter (Too High) ---
  {
    Timer time;
    cloud_data.clear();
    // Same X,Y but Z=3.0 (Max Height is 2.0)
    addPointToCloud(cloud_data, 0.7f, 0.0f, 3.0f);

    float result = run_pc_check(true);
    BOOST_TEST(result == 1.0, "High point (>max_z) should be ignored");

    if (result == 1.0)
      LOG_INFO("Test11 PASSED: PointCloud Height Filter");
  }

  // --- Test 12: Slowdown Zone ---
  {
    Timer time;
    cloud_data.clear();
    // x=0.95. Dist to robot surface = 0.95 - 0.5 = 0.45
    // Slowdown range [0.3, 0.6]. 0.45 is middle -> ~0.5 factor.
    addPointToCloud(cloud_data, 0.95f, 0.0f, 0.5f);

    float result = run_pc_check(true);
    BOOST_TEST((result > 0.4 && result < 0.6),
               "Point in slowdown zone should return approx 0.5, returned " << result);

    if (result > 0.4 && result < 0.6)
      LOG_INFO("Test12 PASSED: PointCloud Slowdown Factor: ", result);
  }

  // --- Test 13: More complex point cloud data ---
  {
    Timer time;
    cloud_data.clear();
    // x=0.95. Dist to robot surface = 0.95 - 0.5 = 0.45
    // Slowdown points: range [0.3, 0.6]. 0.45 is middle -> ~0.5 factor.
    addPointToCloud(cloud_data, 0.95f, 0.0f, 0.5f);
    addPointToCloud(cloud_data, 1.0f, 1.0f, 0.5f);
    addPointToCloud(cloud_data, -1.0f, -1.0f, 0.5f);
    // points to be discarded
    addPointToCloud(cloud_data, -0.1f, -0.1f, 3.0f);
    addPointToCloud(cloud_data, -0.1f, -0.1f, -3.0f);
    addPointToCloud(cloud_data, 0.1f, 0.2f, 4.0f);
    addPointToCloud(cloud_data, 0.1f, 0.2f, -4.0f);
    // Stop points
    addPointToCloud(cloud_data, 0.75f, 0.0f, 0.5f);

    float result = run_pc_check(true);
    BOOST_TEST((result == 0),
               "Point in stop zone should return 0, returned "
                   << result);

    if (result == 0)
      LOG_INFO("Test13 PASSED: Complex PointCloud STOP");
  }

  // --- Test 14: More complex point cloud data - slowdown ---
  {
    Timer time;
    cloud_data.clear();
    // x=0.95. Dist to robot surface = 0.95 - 0.5 = 0.45
    // Slowdown points: range [0.3, 0.6]. 0.45 is middle -> ~0.5 factor.
    addPointToCloud(cloud_data, 0.95f, 0.0f, 0.5f);
    addPointToCloud(cloud_data, -0.95f, 0.0f, 0.5f);
    addPointToCloud(cloud_data, 1.0f, 1.0f, 0.5f);
    addPointToCloud(cloud_data, -1.0f, -1.0f, 0.5f);
    // points to be discarded
    addPointToCloud(cloud_data, -0.1f, -0.1f, 3.0f);
    addPointToCloud(cloud_data, -0.1f, -0.1f, -3.0f);
    addPointToCloud(cloud_data, 0.1f, 0.2f, 4.0f);
    addPointToCloud(cloud_data, 0.1f, 0.2f, -4.0f);

    float result = run_pc_check(false);
    BOOST_TEST((result > 0.4 && result < 0.6),
               "Point in slowdown zone should return approx 0.5, returned "
                   << result);

    if (result > 0.4 && result < 0.6)
      LOG_INFO("Test14 PASSED: PointCloud Slowdown Factor: ", result);
  }
}

// ===========================================================================
//      MULTI-SENSOR POINT CLOUD TESTS (15-18)
// ===========================================================================

namespace {

const auto MS_SHAPE = CollisionChecker::ShapeType::CYLINDER;
const std::vector<float> MS_DIMS{0.51f, 2.0f};
constexpr float MS_CRIT_ANGLE = 160.0f;
constexpr float MS_CRIT_DIST = 0.3f;
constexpr float MS_SLOW_DIST = 0.6f;
constexpr float MS_MIN_H = 0.1f; // body frame
constexpr float MS_MAX_H = 2.0f; // body frame

const int MS_POINT_STEP = sizeof(PointXYZ);
const int MS_X_OFF = offsetof(PointXYZ, x);
const int MS_Y_OFF = offsetof(PointXYZ, y);
const int MS_Z_OFF = offsetof(PointXYZ, z);

PointCloudView ms_view(const std::vector<uint8_t> &cloud) {
  const int n = static_cast<int>(cloud.size() / MS_POINT_STEP);
  return PointCloudView{cloud, MS_POINT_STEP, n * MS_POINT_STEP,
                        n > 0 ? 1 : 0, n,
                        MS_X_OFF,      MS_Y_OFF, MS_Z_OFF};
}

CriticalZoneChecker make_ms_checker(const std::vector<SensorConfig> &sensors) {
  return CriticalZoneChecker(CriticalZoneChecker::InputType::POINTCLOUD,
                             MS_SHAPE, MS_DIMS, sensors, MS_CRIT_ANGLE,
                             MS_CRIT_DIST, MS_SLOW_DIST, MS_MIN_H, MS_MAX_H,
                             20.0f);
}

std::vector<SensorConfig> ms_front_back() {
  return {SensorConfig::fromYaw({0.2f, 0.0f, 0.2f}, 0.0f),
          SensorConfig::fromYaw({-0.2f, 0.0f, 0.2f},
                                static_cast<float>(M_PI))};
}

} // namespace

// --- Test 15: Back-Sensor Obstacle & Moving Backward (Headline Case) ---
BOOST_AUTO_TEST_CASE(test_cpu_multi_back_obstacle_stops_reverse) {
  Timer time;
  auto checker = make_ms_checker(ms_front_back());
  std::vector<uint8_t> back_cloud;
  // Obstacle seen ONLY by the back sensor (mounted at x=-0.2, yaw=pi).
  // Back-sensor frame (0.55, 0, 0.3) -> body (-0.75, 0, 0.5).
  // Dist = 0.75. 0.75 - Radius(0.51) = 0.24 < Crit(0.3) -> Critical.
  addPointToCloud(back_cloud, 0.55f, 0.0f, 0.3f);
  const std::vector<uint8_t> empty_front;

  const float backward = checker.check(
      {ms_view(empty_front), ms_view(back_cloud)}, /*forward*/ false);
  BOOST_TEST(backward == 0.0f,
             "Obstacle is behind and robot is moving backward -> Critical "
             "zone result should be 0.0, returned "
                 << backward);

  const float forward = checker.check(
      {ms_view(empty_front), ms_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST(forward == 1.0f,
             "Obstacle is behind and robot is moving forward -> Critical "
             "zone result should be 1.0, returned "
                 << forward);

  if (backward == 0.0f && forward == 1.0f) {
    LOG_INFO("Test15 PASSED: Back-sensor obstacle stops reverse motion and "
             "does not affect forward motion");
  }
}

// --- Test 16: Min-Wins Fusion Across Clouds ---
BOOST_AUTO_TEST_CASE(test_cpu_multi_min_across_clouds) {
  Timer time;
  auto checker = make_ms_checker(ms_front_back());
  std::vector<uint8_t> front_cloud;
  // Front-sensor frame (0.75, 0, 0.3) -> body (0.95, 0, 0.5).
  // Dist to robot = 0.95 - 0.51 = 0.44. Slowdown range [0.3, 0.6] ->
  // factor (0.44 - 0.3) / 0.3 = ~0.467.
  addPointToCloud(front_cloud, 0.75f, 0.0f, 0.3f);
  std::vector<uint8_t> back_cloud;
  // Far point in the back cloud -> safe, must not affect the fused result
  addPointToCloud(back_cloud, 5.0f, 0.0f, 0.3f);

  const float slow = checker.check(
      {ms_view(front_cloud), ms_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST((slow > 0.4f && slow < 0.55f),
             "Front slowdown point + far back point -> fused factor should "
             "be ~0.467, returned "
                 << slow);

  // Front-sensor frame (0.6, 0, 0.3) -> body (0.8, 0, 0.5).
  // 0.8 - Radius(0.51) = 0.29 < Crit(0.3) -> Critical, must win the min.
  addPointToCloud(front_cloud, 0.6f, 0.0f, 0.3f);
  const float stop = checker.check(
      {ms_view(front_cloud), ms_view(back_cloud)}, /*forward*/ true);
  BOOST_TEST(stop == 0.0f,
             "Critical point should win the min across clouds -> result "
             "should be 0.0, returned "
                 << stop);

  // All-empty batch -> nothing observed this tick -> no constraint
  const std::vector<uint8_t> empty;
  const float idle =
      checker.check({ms_view(empty), ms_view(empty)}, /*forward*/ true);
  BOOST_TEST(idle == 1.0f,
             "All-empty batch -> result should be 1.0, returned " << idle);

  if (slow > 0.4f && slow < 0.55f && stop == 0.0f && idle == 1.0f) {
    LOG_INFO("Test16 PASSED: Min factor wins across fused clouds, slowdown "
             "factor = ",
             slow);
  }
}

// --- Test 17: Tilted Mount, Exact Values ---
BOOST_AUTO_TEST_CASE(test_cpu_multi_tilted_mount_exact) {
  Timer time;
  // 25 deg pitch mount at z=0.3 (quaternion built from the half angle)
  const float half_pitch = 12.5f * static_cast<float>(M_PI) / 180.0f;
  SensorConfig tilted;
  tilted.position = {0.0f, 0.0f, 0.3f};
  tilted.rotation = {0.0f, std::sin(half_pitch), 0.0f, std::cos(half_pitch)};
  auto checker = make_ms_checker({tilted});

  std::vector<uint8_t> cloud;
  // Sensor frame (0.8, 0, 0.3): the full transform leans z INTO x:
  // x_body = 0.8*cos25 + 0.3*sin25 = 0.852 -> slowdown factor ~0.14.
  // Dropping the transform's z column (the old approximation) gives
  // x_body = 0.725 -> a false STOP, so a 0 here means the tilt terms
  // regressed.
  addPointToCloud(cloud, 0.8f, 0.0f, 0.3f);
  const float factor = checker.check({ms_view(cloud)}, /*forward*/ true);
  BOOST_TEST((factor > 0.05f && factor < 0.25f),
             "Tilted mount -> slowdown factor should be ~0.14 under the "
             "full transform, returned "
                 << factor);

  if (factor > 0.05f && factor < 0.25f) {
    LOG_INFO("Test17 PASSED: Full tilt transform applied, slowdown factor = ",
             factor);
  }
}

// --- Test 18: N=1 Adapter & Error Paths ---
BOOST_AUTO_TEST_CASE(test_cpu_multi_adapter_and_errors) {
  Timer time;
  auto single = make_ms_checker({SensorConfig{}});
  std::vector<uint8_t> cloud;
  addPointToCloud(cloud, 0.95f, 0.0f, 0.5f);
  addPointToCloud(cloud, 0.75f, 0.0f, 0.5f);

  // The single-cloud byte entry is an N=1 adapter onto the batched
  // path -> identical result for the identical cloud
  const auto view = ms_view(cloud);
  const float legacy =
      single.check(view.data, view.point_step, view.row_step, view.height,
                   view.width, view.x_offset, view.y_offset, view.z_offset,
                   /*forward*/ true);
  const float batched = single.check({view}, /*forward*/ true);
  BOOST_TEST(legacy == batched,
             "Adapter and batched entry should agree, returned "
                 << legacy << " vs " << batched);

  // Cloud count mismatch: two configured sensors, one cloud
  auto multi = make_ms_checker(ms_front_back());
  BOOST_CHECK_THROW(multi.check({view}, true), std::invalid_argument);

  // Malformed metadata must name the offending cloud in the error message
  PointCloudView bad = ms_view(cloud);
  bad.y_offset = -4;
  BOOST_CHECK_EXCEPTION(
      multi.check({ms_view(cloud), bad}, true), std::invalid_argument,
      [](const std::invalid_argument &e) {
        return std::string(e.what()).find("clouds[1]") != std::string::npos &&
               std::string(e.what()).find("non-negative") != std::string::npos;
      });

  // Laserscan entry on a pointcloud-mode checker throws
  Eigen::VectorXf ranges(4);
  ranges.setConstant(1.0f);
  BOOST_CHECK_THROW(multi.check(ranges, true), std::logic_error);

  if (legacy == batched) {
    LOG_INFO("Test18 PASSED: N=1 adapter matches batched entry and error "
             "paths throw as expected");
  }
}

// --- Test 19: Optical-Axis Point Is Not Filtered (Depth Camera) ---
BOOST_AUTO_TEST_CASE(test_cpu_multi_optical_axis_not_filtered) {
  Timer time;
  // Depth camera in the ROS optical frame (z forward, x right, y down),
  // mounted at (0.1, 0, 0.3). Body-from-optical quaternion [x,y,z,w] =
  // [-0.5, 0.5, -0.5, 0.5]: x_body = z_opt, y_body = -x_opt, z_body = -y_opt
  SensorConfig camera;
  camera.position = {0.1f, 0.0f, 0.3f};
  camera.rotation = {-0.5f, 0.5f, -0.5f, 0.5f};
  auto checker = make_ms_checker({camera});

  std::vector<uint8_t> cloud;
  // Sensor (0, 0, 0.4) lies ON the optical axis -> body (0.5, 0, 0.3).
  // Dist = 0.5. 0.5 - Radius(0.51) = -0.01 < Crit(0.3) -> Critical. A
  // planar sensor-frame origin filter (x^2+y^2 < 1e-6) deletes this point
  // -> full speed into an obstacle 40 cm dead ahead of the camera
  addPointToCloud(cloud, 0.0f, 0.0f, 0.4f);
  const float factor = checker.check({ms_view(cloud)}, /*forward*/ true);
  BOOST_TEST(factor == 0.0f,
             "On-axis point 40 cm ahead of a depth camera -> Critical zone "
             "result should be 0.0, returned "
                 << factor);

  // A TRUE origin point (all three coordinates zero) must still be
  // rejected: it maps onto the mount position inside the robot
  std::vector<uint8_t> origin_cloud;
  addPointToCloud(origin_cloud, 0.0f, 0.0f, 0.0f);
  const float clear = checker.check({ms_view(origin_cloud)}, /*forward*/ true);
  BOOST_TEST(clear == 1.0f,
             "Sensor-origin self-return -> Critical zone result should be "
             "1.0, returned "
                 << clear);

  if (factor == 0.0f && clear == 1.0f) {
    LOG_INFO("Test19 PASSED: Optical-axis obstacle detected and origin "
             "self-return filtered");
  }
}
