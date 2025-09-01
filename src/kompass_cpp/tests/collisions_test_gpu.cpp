#include "test.h"
#include <Eigen/Dense>
#define BOOST_TEST_MODULE KOMPASS TESTS
#include "collisions_test.cpp"
#include "utils/collision_check.h"
#include "utils/critical_zone_check_gpu.h"
#include "utils/logger.h"
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

BOOST_AUTO_TEST_CASE(test_critical_zone_check_gpu) {
  // Create timer
  Timer time;

  auto robotShapeType = CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.51, 0.27, 0.4};

  const Eigen::Vector3f sensor_position_body{0.22, 0.0, 0.4};
  const Eigen::Vector4f sensor_rotation_body{0, 0, 0.99, 0.0};

  // Robot laserscan value
  std::vector<double> scan_angles;
  std::vector<double> scan_ranges;
  initLaserscan(360, 10.0, scan_ranges, scan_angles);

  bool forward_motion = true;
  float critical_angle = 160.0, critical_distance = 0.3,
        slowdown_distance = 0.6;

  CriticalZoneCheckerGPU zoneChecker(
      robotShapeType, robotDimensions, sensor_position_body,
      sensor_rotation_body, critical_angle, critical_distance,
      slowdown_distance, scan_angles, 0.0, 0.0, 20.0);

  LOG_INFO("Testing Emergency Stop with GPU ");

  // Set small ranges behind the robot
  setLaserscanAtAngle(0.0, 0.2, scan_ranges, scan_angles);
  setLaserscanAtAngle(0.1, 0.2, scan_ranges, scan_angles);
  setLaserscanAtAngle(-0.1, 0.2, scan_ranges, scan_angles);
  float result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(result == 1.0, "Angles are behind and robot is moving forward -> "
                            "Critical zone result should be 1.0, returned "
                                << result);
  if (result == 1.0) {
    LOG_INFO("Test1 PASSED: Angles are behind and robot is moving forward");
  }

  // Set small ranges behind the robot
  initLaserscan(360, 10.0, scan_ranges, scan_angles);
  result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(result == 1.0,
             "Angles are in front and far and robot is moving forward "
             "-> Critical zone result should be 1.0, returned "
                 << result);
  if (result == 1.0) {
    LOG_INFO("Test2 PASSED: Angles are in front and robot is moving forward");
  }

  // Set small ranges in front of the robot
  setLaserscanAtAngle(M_PI, 0.2, scan_ranges, scan_angles);
  setLaserscanAtAngle(M_PI + 0.1, 0.2, scan_ranges, scan_angles);
  setLaserscanAtAngle(M_PI - 0.1, 0.2, scan_ranges, scan_angles);
  result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(result == 0.0,
             "Angles are in front and close and robot is moving "
             "forward -> Critical zone result should be 0.0, returned "
                 << result);
  if (result == 0.0) {
    LOG_INFO("Test3 PASSED: Angles are in front and close and robot is moving "
             "forward");
  }

  forward_motion = false;
  result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(result == 1.0,
             "Angles are in front and close and robot is moving "
             "backwards-> Critical zone result should be 1.0, returned "
                 << result);
  if (result == 1.0) {
    LOG_INFO("Test4 PASSED: Angles are in front and close and robot is "
             "moving backward");
  }

  // Set small ranges behind the robot
  setLaserscanAtAngle(0.0, 0.2, scan_ranges, scan_angles);
  setLaserscanAtAngle(0.1, 0.2, scan_ranges, scan_angles);
  setLaserscanAtAngle(-0.1, 0.2, scan_ranges, scan_angles);
  result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(result == 0.0,
             "Angles are in back and close and robot is moving "
             "backwards -> Critical zone result should be 0.0, returned "
                 << result);
  if (result == 0.0) {
    LOG_INFO("Test5 PASSED: Angles are in back and close and robot is moving "
             "backwards");
  }

  // Set slowdown ranges behind the robot
  forward_motion = false;
  initLaserscan(360, 10.0, scan_ranges, scan_angles);
  setLaserscanAtAngle(0.0, 1.0, scan_ranges, scan_angles);
  result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(
      (result > 0.0f and result < 1.0f),
      "Angles are in back and in the slowdown zone and robot is moving "
      "backwards -> Critical zone result should be between [0, 1], returned "
          << result);
  if (result > 0.0f and result < 1.0f) {
    LOG_INFO("Test6 PASSED: Angles are in back and in the slowdown zone and "
             "robot is moving "
             "backwards, slowdown factor = ",
             result);
  }

  forward_motion = true;
  result = zoneChecker.check(scan_ranges, forward_motion);
  BOOST_TEST(result == 1.0,
             "Angles are in back and in the slowdown zone and robot is moving "
             "forward -> Critical zone result should be between 1.0, returned "
                 << result);
  if (result == 1.0) {
    LOG_INFO("Test7 PASSED: Angles are in back and in the slowdown zone and "
             "robot is moving "
             "forward, slowdown factor = ",
             result);
  }

  // Set slowdown ranges infront of the robot
  setLaserscanAtAngle(M_PI, 0.5, scan_ranges, scan_angles);
  setLaserscanAtAngle(M_PI + 0.1, 0.4, scan_ranges, scan_angles);
  setLaserscanAtAngle(M_PI - 0.1, 0.4, scan_ranges, scan_angles);
  result = zoneChecker.check(scan_ranges, forward_motion);
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
