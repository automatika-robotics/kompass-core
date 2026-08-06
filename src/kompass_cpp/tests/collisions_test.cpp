#define BOOST_TEST_MODULE KOMPASS_COLLISION_TESTS
#include "test.h"
#include "utils/collision_check.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include <vector>

using namespace Kompass;

BOOST_AUTO_TEST_CASE(test_FCL) {
  // Shared setup
  double octreeRes = 0.1;
  auto robotShapeType = CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.4, 0.4, 1.0};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 1.0};
  const Eigen::Quaternionf sensor_rotation_body{0, 0, 0, 1};

  CollisionChecker collChecker(robotShapeType, robotDimensions,
                               sensor_position_body, sensor_rotation_body,
                               octreeRes);

  LOG_INFO("Testing collision checker using Laserscan data");

  // --- Block 1: Test False Collision ---
  {
    Timer time;
    Path::State robotState(0.0, 0.0, 0.0, 0.0);
    Eigen::VectorXf scan_angles = toVecF({0.0, 0.1, 0.2});
    Eigen::VectorXf scan_ranges = toVecF({1.0, 1.0, 1.0});

    LOG_INFO("Running test with robot at ", robotState.x, ", ", robotState.y);
    LOG_INFO("Scan Ranges ", scan_ranges[0], ", ", scan_ranges[1], ", ",
             scan_ranges[2]);

    collChecker.updateState(robotState);
    bool res_false = collChecker.checkCollisions(scan_ranges, scan_angles);
    BOOST_TEST(!res_false,
               "Collision Result should be FALSE Got: " << res_false);
    std::cout << std::endl;
  }

  // --- Block 2: Test True Collision ---
  {
    Timer time;
    Path::State robotState(3.0, 5.0, 0.0, 0.0);
    // Reuse angles from previous scope or define new ones if needed.
    // Since scan_angles was local to Block 1, we redefine them here for
    // clarity/isolation.
    Eigen::VectorXf scan_angles = toVecF({0.0, 0.1, 0.2});
    Eigen::VectorXf scan_ranges = toVecF({0.25, 0.5, 0.5});

    LOG_INFO("Running test with robot at ", robotState.x, ", ", robotState.y);
    LOG_INFO("Scan Ranges ", scan_ranges[0], ", ", scan_ranges[1], ", ",
             scan_ranges[2]);

    collChecker.updateState(robotState);
    bool res_true = collChecker.checkCollisions(scan_ranges, scan_angles);
    BOOST_TEST(res_true, "Collision Result should be TRUE got: " << res_true);
    std::cout << std::endl;
  }

  // --- Block 3: Pointcloud Collision ---
  {
    Timer time;
    Path::State robotState(3.0, 5.0, 0.0, 0.0);

    LOG_INFO("Testing collision between: \nRobot at {x: ", robotState.x,
             ", y: ", robotState.y, "}\n", "and Pointcloud");

    // Point clouds are taken in the WORLD frame: this obstacle sits just
    // inside the robot box centred at (3, 5), no transform applied.
    std::vector<Path::Point> cloud;
    cloud.push_back(Path::Point(3.1, 5.1, -0.5));

    // Place the body explicitly rather than inheriting Block 2's pose
    collChecker.updateState(robotState);
    collChecker.updateSensorData(cloud, robotState);
    bool res = collChecker.checkCollisions();
    BOOST_TEST(res, "Collision Result: " << res);
  }
}
