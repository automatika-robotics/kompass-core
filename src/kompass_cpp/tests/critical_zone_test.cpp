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
  auto robotShapeType = CollisionChecker::ShapeType::BOX;
  CriticalZoneChecker::InputType inputType =
      CriticalZoneChecker::InputType::LASERSCAN;
  std::vector<float> robotDimensions{0.51, 0.27, 0.4};

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
      inputType, robotShapeType, robotDimensions, sensor_position_body,
      sensor_rotation_body, critical_angle, critical_distance,
      slowdown_distance, scan_angles, min_height, max_height, 20.0);

  LOG_INFO("Testing Emergency Stop with CPU (LASERSCAN)");

  // --- Test 1: Behind & Moving Forward ---
  {
    Timer time;
    bool forward_motion = true;
    setLaserscanAtAngle(0.0, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(0.1, 0.2, scan_ranges, scan_angles);
    setLaserscanAtAngle(-0.1, 0.2, scan_ranges, scan_angles);

    float result = zoneChecker.check(scan_ranges, forward_motion);
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

    float result = zoneChecker.check(scan_ranges, forward_motion);
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

    float result = zoneChecker.check(scan_ranges, forward_motion);
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

    float result = zoneChecker.check(scan_ranges, forward_motion);
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

    float result = zoneChecker.check(scan_ranges, forward_motion);
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
    setLaserscanAtAngle(0.0, 1.0, scan_ranges, scan_angles);

    float result = zoneChecker.check(scan_ranges, forward_motion);
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

    float result = zoneChecker.check(scan_ranges, forward_motion);
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
    setLaserscanAtAngle(M_PI, 0.5, scan_ranges, scan_angles);
    setLaserscanAtAngle(M_PI + 0.1, 0.4, scan_ranges, scan_angles);
    setLaserscanAtAngle(M_PI - 0.1, 0.4, scan_ranges, scan_angles);

    float result = zoneChecker.check(scan_ranges, forward_motion);
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

  // Instantiate a separate checker for PointCloud mode
  auto pcShapeType = CollisionChecker::ShapeType::SPHERE;
  std::vector<float> pcDims{0.5}; // Radius = 0.5
  Eigen::Vector3f pcPos{0.0, 0.0, 0.0};
  Eigen::Vector4f pcRot{0.0, 0.0, 0.0, 1.0};

  std::vector<double> pc_angles;
  std::vector<double> dummy_ranges;
  initLaserscan(360, 10.0, dummy_ranges, pc_angles); // 1-degree resolution

  CriticalZoneChecker pcChecker(
      CriticalZoneChecker::InputType::POINTCLOUD, pcShapeType, pcDims, pcPos,
      pcRot, critical_angle, 0.5 /*crit_dist*/, 1.0 /*slow_dist*/,
      pc_angles, // Size 360
      0.1 /*min_h*/, 2.0 /*max_h*/, 20.0);

  std::vector<int8_t> cloud_data;

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
    // Dist = 0.8. RobotRad(0.5) + Crit(0.5) = 1.0. -> 0.8 < 1.0 -> Critical.
    addPointToCloud(cloud_data, 0.8f, 0.0f, 0.5f);

    float result = run_pc_check(true);
    BOOST_TEST(result == 0.0, "Point at 0.8m should trigger stop (0.0)");

    if (result == 0.0)
      LOG_INFO("Test10 PASSED: PointCloud Critical Stop");
  }

  // --- Test 11: Height Filter (Too High) ---
  {
    Timer time;
    cloud_data.clear();
    // Same X,Y but Z=3.0 (Max Height is 2.0)
    addPointToCloud(cloud_data, 0.8f, 0.0f, 3.0f);

    float result = run_pc_check(true);
    BOOST_TEST(result == 1.0, "High point (>max_z) should be ignored");

    if (result == 1.0)
      LOG_INFO("Test11 PASSED: PointCloud Height Filter");
  }

  // --- Test 12: Slowdown Zone ---
  {
    Timer time;
    cloud_data.clear();
    // x=1.25. Dist to robot surface = 1.25 - 0.5 = 0.75.
    // Slowdown range [0.5, 1.0]. 0.75 is middle -> ~0.5 factor.
    addPointToCloud(cloud_data, 1.25f, 0.0f, 0.5f);

    float result = run_pc_check(true);
    BOOST_TEST((result > 0.4 && result < 0.6),
               "Point in slowdown zone should return approx 0.5");

    if (result > 0.4 && result < 0.6)
      LOG_INFO("Test12 PASSED: PointCloud Slowdown Factor: ", result);
  }
}
