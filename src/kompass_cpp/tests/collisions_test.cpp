#include "test.h"
#define BOOST_TEST_MODULE KOMPASS TESTS
#include "utils/collision_check.h"
#include "utils/critical_zone_check.h"
#include "utils/logger.h"
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass;

BOOST_AUTO_TEST_CASE(test_FCL) {
  // Create timer
  Timer time;

  // Octomap resolution
  double octreeRes = 0.1;

  auto robotShapeType = CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.4, 0.4, 1.0};

  const std::array<float, 3> sensor_position_body{0.0, 0.0, 1.0};
  const std::array<float, 4> sensor_rotation_body{0, 0, 0, 1};

  // Robot start state (pose)
  Path::State robotState(0.0, 0.0, 0.0, 0.0);

  // Robot laserscan value (empty)
  std::vector<double> scan_angles{0, 0.1, 0.2};
  std::vector<double> scan_ranges{1.0, 1.0, 1.0};

  CollisionChecker collChecker(robotShapeType, robotDimensions,
                               sensor_position_body, sensor_rotation_body,
                               octreeRes);

  LOG_INFO("Testing collision checker using Laserscan data");

  collChecker.updateState(robotState);

  bool res_false = collChecker.checkCollisions(scan_ranges, scan_angles);
  LOG_INFO("Testing collision between: \nRobot at {x: ", robotState.x,
           ", y: ", robotState.y, "}\n",
           "and Laserscan with: ranges {1.0, 1.0, 1.0, 1.0} at angles {0, 0.1, "
           "0.2, 3.14}, Collision: ",
           res_false);
  BOOST_TEST(!res_false, "Non Collision Result: " << res_false);

  robotState.x = 3.0;
  robotState.y = 5.0;
  collChecker.updateState(robotState);

  scan_ranges = {0.16, 0.5, 0.5};

  bool res_true = collChecker.checkCollisions(scan_ranges, scan_angles);
  LOG_INFO("Testing collision between: \nRobot at {x: ", robotState.x,
           ", y: ", robotState.y, "}\n",
           "and Laserscan with: ranges {0.2, 0.5, 0.5} at angles {0, 0.1, "
           "0.2, 3.14} -> Collision: ",
           res_true);
  BOOST_TEST(res_true, "Collision Result: " << res_true);

  LOG_INFO("Testing collision between: \nRobot at {x: ", robotState.x,
           ", y: ", robotState.y, "}\n", "and Pointcloud");
  std::vector<Path::Point> cloud;
  // Point cloud in sensor frame
  cloud.push_back(Path::Point(0.1, 0.1, -0.5));
  collChecker.updatePointCloud(cloud);
  bool res = collChecker.checkCollisions();
  float dist = collChecker.getMinDistance();
  LOG_INFO("Min distance is: ", dist);
  BOOST_TEST((dist <= 0.0), "Min distance <= 0 " << (dist <= 0.0));
  BOOST_TEST(res, "Collision Result: " << res);
}

BOOST_AUTO_TEST_CASE(test_critical_zone_check) {
  // Create timer
  Timer time;

  auto robotShapeType = CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.51, 0.27, 0.4};

  const std::array<float, 3> sensor_position_body{0.22, 0.0, 0.4};
  const std::array<float, 4> sensor_rotation_body{0, 0, 0.99, 0.0};

  // Robot laserscan value
  std::vector<double> scan_angles{0, 0.1, 0.2};
  std::vector<double> scan_ranges{1.0, 1.0, 1.0};
  bool forward_motion = true;
  float critical_angle = 120.0, critical_distance = 0.2;

  CriticalZoneChecker zoneChecker(robotShapeType, robotDimensions,
                                  sensor_position_body, sensor_rotation_body,
                                  critical_angle, critical_distance);

  LOG_INFO("Testing Emergency Stop with CPU");

  bool result = zoneChecker.check(scan_ranges, scan_angles, forward_motion);
  BOOST_TEST(!result, "Angles are behind and robot is moving forward -> "
                      "Critical zone result should be FALSE, returned"
                          << result);
  if (!result) {
    LOG_INFO("Test1 PASSED: Angles are behind and robot is moving forward");
  }

  scan_angles = {M_PI, M_PI + 0.2, M_PI - 0.2};
  scan_ranges = {1.0, 1.0, 1.0};
  result = zoneChecker.check(scan_ranges, scan_angles, forward_motion);
  BOOST_TEST(!result, "Angles are in front and far and robot is moving forward "
                      "-> Critical zone result should be FALSE, returned "
                          << result);
  if (!result) {
    LOG_INFO("Test2 PASSED: Angles are in front and robot is moving forward");
  }

  scan_ranges = {0.5, 0.3, 0.6};
  result = zoneChecker.check(scan_ranges, scan_angles, forward_motion);
  BOOST_TEST(result, "Angles are in front and close and robot is moving "
                     "forward -> Critical zone result should be TRUE, returned "
                         << result);
  if (result) {
    LOG_INFO("Test3 PASSED: Angles are in front and close and robot is moving "
             "forward");
  }

  forward_motion = false;
  result = zoneChecker.check(scan_ranges, scan_angles, forward_motion);
  BOOST_TEST(!result,
             "Angles are in front and close and robot is moving "
             "backwards-> Critical zone result should be FALSE, returned "
                 << result);
  if (!result) {
    LOG_INFO("Test4 PASSED: Angles are in front and close and robot is "
             "moving backward");
  }

  scan_angles = {0.0, 0.2, -0.2};
  result = zoneChecker.check(scan_ranges, scan_angles, forward_motion);
  BOOST_TEST(result,
             "Angles are in back and close and robot is moving "
             "backwards -> Critical zone result should be TRUE, returned "
                 << result);
  if (result) {
    LOG_INFO("Test5 PASSED: Angles are in back and close and robot is moving "
             "backwards");
  }
}
