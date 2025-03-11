#include "controllers/dwa.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#define BOOST_TEST_MODULE KOMPASS TESTS
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass;

void applyControl(Path::State &robotState, const Control::Velocity2D control,
                  const double timeStep) {
  double dx = (control.vx() * std::cos(robotState.yaw) -
               control.vy() * std::sin(robotState.yaw)) *
              timeStep;
  double dy = (control.vx() * std::sin(robotState.yaw) +
               control.vy() * std::cos(robotState.yaw)) *
              timeStep;
  double dyaw = control.omega() * timeStep;
  robotState.x += dx;
  robotState.y += dy;
  robotState.yaw += dyaw;
}

BOOST_AUTO_TEST_CASE(test_DWA) {
  // Create timer
  Timer time;

  // Create a test path
  std::vector<Path::Point> points{Path::Point(0.0, 0.0), Path::Point(1.0, 0.0),
                                  Path::Point(2.0, 0.0)};
  Path::Path path(points);

  // Sampling configuration
  double timeStep = 0.1;
  double predictionHorizon = 4.0;
  double controlHorizon = 0.4;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;
  int maxNumThreads = 10;

  // Octomap resolution
  double octreeRes = 0.1;

  // Cost weights
  Control::CostEvaluator::TrajectoryCostsWeights costWeights;
  costWeights.setParameter("reference_path_distance_weight", 1.0);
  costWeights.setParameter("goal_distance_weight", 3.0);

  // Robot configuration
  Control::LinearVelocityControlParams x_params(1, 3, 5);
  Control::LinearVelocityControlParams y_params(1, 3, 5);
  Control::AngularVelocityControlParams angular_params(3.14, 3, 5, 8);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);
  auto controlType = Control::ControlType::ACKERMANN;
  auto robotShapeType = Kompass::CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.3, 0.3, 1.0};
  // std::array<float, 3> sensorPositionWRTbody {0.0, 0.0, 1.0};
  const std::array<float, 3> sensor_position_body{0.0, 0.0, 0.5};
  const std::array<float, 4> sensor_rotation_body{0, 0, 0, 1};

  // Robot start state (pose)
  Path::State robotState(0.0, 0.0, 0.0, 0.0);

  // Robot initial velocity control
  Control::Velocity2D robotControl;

  // Robot laserscan value (empty)
  Control::LaserScan robotScan({10.0, 10.0, 10.0}, {0, 0.1, 0.2});

  LOG_INFO("Setting up DWA planner");

  Control::DWA planner(controlLimits, controlType, timeStep, predictionHorizon,
                       controlHorizon, maxLinearSamples, maxAngularSamples,
                       robotShapeType, robotDimensions, sensor_position_body,
                       sensor_rotation_body, octreeRes, costWeights,
                       maxNumThreads);

  LOG_INFO("Simulating one step of DWA planner");

  // Set the global path in the planner
  planner.setCurrentPath(path);

  int counter = 0;

  while (!planner.isGoalReached() and counter < 100) {
    counter++;
    // Set the robot state in the planner
    planner.setCurrentState(robotState);

    Control::TrajSearchResult result =
        planner.computeVelocityCommandsSet(robotControl, robotScan);

    LOG_INFO(FMAG("Result Found: "), result.isTrajFound);

    auto current_path = planner.getCurrentPath();

    LOG_DEBUG(FMAG("Current segment: "), planner.getCurrentSegmentIndex(),
              FMAG(", Max segments: "), current_path.getMaxNumSegments());

    if (result.isTrajFound) {
      LOG_INFO(FMAG("Found best trajectory with cost: "), result.trajCost);

      double vx = planner.getLinearVelocityCmdX();
      double vy = planner.getLinearVelocityCmdY();
      double omega = planner.getAngularVelocityCmd();

      LOG_DEBUG(BOLD(FBLU("Robot at: {x: ")), KBLU, robotState.x, RST,
                BOLD(FBLU(", y: ")), KBLU, robotState.y, RST,
                BOLD(FBLU(", yaw: ")), KBLU, robotState.yaw, RST,
                BOLD(FBLU("}")));

      LOG_DEBUG(BOLD(FGRN("Found Control: {Vx: ")), KGRN, vx, RST,
                BOLD(FGRN(", Vy: ")), KGRN, vy, RST, BOLD(FGRN(", Omega: ")),
                KGRN, omega, RST, BOLD(FGRN("}")));

      applyControl(robotState, Control::Velocity2D(vx, vy, omega), timeStep);
    } else {
      LOG_ERROR(BOLD(FRED("DWA Planner failed with robot at: {x: ")), KRED,
                robotState.x, RST, BOLD(FRED(", y: ")), KRED, robotState.y, RST,
                BOLD(FRED(", yaw: ")), KRED, robotState.yaw, RST,
                BOLD(FRED("}")));
    }
  }

  BOOST_TEST(planner.isGoalReached(),
             "Goal Reached in " << counter << " steps");
}

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

  // Octomap resolution
  double octreeRes = 0.1;

  auto robotShapeType = CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.51, 0.27, 0.4};

  const std::array<float, 3> sensor_position_body{0.22, 0.0, 0.4};
  const std::array<float, 4> sensor_rotation_body{0, 0, 0.99, 0.0};

  // Robot start state (pose)
  Path::State robotState(0.0, 0.0, 0.0, 0.0);

  // Robot laserscan value (empty)
  std::vector<double> scan_angles{0, 0.1, 0.2};
  std::vector<double> scan_ranges{1.0, 1.0, 1.0};
  bool forward_motion = true;
  float critical_angle = 120.0, critical_distance = 0.2;

  CollisionChecker collChecker(robotShapeType, robotDimensions,
                               sensor_position_body, sensor_rotation_body,
                               octreeRes);

  LOG_INFO("Testing Emergency Stop");

  collChecker.updateState(robotState);

  bool result =
      collChecker.checkCriticalZone(scan_ranges, scan_angles, forward_motion,
                                    critical_angle, critical_distance);
  BOOST_TEST(
      !result,
      "Angles are behind and robot is moving forward -> Critical zone result: "
          << result);

  scan_angles = {M_PI, M_PI + 0.2, M_PI - 0.2};
  scan_ranges = {1.0, 1.0, 1.0};
  result =
      collChecker.checkCriticalZone(scan_ranges, scan_angles, forward_motion,
                                    critical_angle, critical_distance);
  BOOST_TEST(!result, "Angles are in front and far and robot is moving forward "
                      "-> Critical zone result: "
                          << result);

  scan_angles = {M_PI, M_PI + 0.2, M_PI - 0.2};
  scan_ranges = {0.5, 0.3, 0.6};
  result =
      collChecker.checkCriticalZone(scan_ranges, scan_angles, forward_motion,
                                    critical_angle, critical_distance);
  BOOST_TEST(result, "Angles are in front and close and robot is moving "
                     "forward -> Critical zone result: "
                         << result);

  forward_motion = false;
  result =
      collChecker.checkCriticalZone(scan_ranges, scan_angles, forward_motion,
                                    critical_angle, critical_distance);
  BOOST_TEST(!result, "Angles are in front and close and robot is moving "
                      "backwards-> Critical zone result: "
                          << result);

  scan_angles = {0.0, 0.2, -0.2};
  scan_ranges = {0.5, 0.3, 0.6};
  result =
      collChecker.checkCriticalZone(scan_ranges, scan_angles, forward_motion,
                                    critical_angle, critical_distance);
  BOOST_TEST(result, "Angles are in back and close and robot is moving "
                     "backwards -> Critical zone result: "
                         << result);
}
