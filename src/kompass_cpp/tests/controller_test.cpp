#include "controllers/dwa.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include <system_error>
#define BOOST_TEST_MODULE KOMPASS TESTS
#include "json_export.h"
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
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
  std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0), Path::Point(1.0, 0.0, 0.0),
                                  Path::Point(2.0, 0.0, 0.0)};
  Path::Path path(points, 500);

  // Sampling configuration
  double timeStep = 0.1;
  double predictionHorizon = 1.0;
  double controlHorizon = 0.2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;
  int maxNumThreads = 10;

  // Octomap resolution
  double octreeRes = 0.1;

  // Cost weights
  Control::CostEvaluator::TrajectoryCostsWeights costWeights;
  costWeights.setParameter("reference_path_distance_weight", 1.0);
  costWeights.setParameter("goal_distance_weight", 3.0);
  costWeights.setParameter("obstacles_distance_weight", 0.0);
  costWeights.setParameter("smoothness_weight", 0.0);
  costWeights.setParameter("jerk_weight", 0.0);

  // Robot configuration
  Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0);
  Control::LinearVelocityControlParams y_params(1, 3, 5);
  Control::AngularVelocityControlParams angular_params(3.14, 2.0, 3.0, 3.0);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);
  auto controlType = Control::ControlType::ACKERMANN;
  auto robotShapeType = Kompass::CollisionChecker::ShapeType::CYLINDER;
  std::vector<float> robotDimensions{0.1, 0.4};
  // std::array<float, 3> sensorPositionWRTbody {0.0, 0.0, 1.0};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 0.0};
  const Eigen::Vector4f sensor_rotation_body{0, 0, 0, 1};

  // Robot start state (pose)
  Path::State robotState(-0.51731912, 0.0, 0.0, 0.0);

  // Robot initial velocity control
  Control::Velocity2D robotControl;

  // Robot laserscan value (empty)
  Control::LaserScan robotScan({0.4, 0.3}, {10, 10.1});

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

  planner.setCurrentState(robotState);

  planner.debugVelocitySearch(robotControl, robotScan, true);

  Control::TrajectorySamples2D samples_ = planner.getDebuggingSamplesPure();

  // Plot the trajectories (Save to json then run python script for plotting)
  boost::filesystem::path executablePath = boost::dll::program_location();
  std::string file_location = executablePath.parent_path().string();

  std::string trajectories_filename =
      file_location + "/trajectories_controller_test";
  std::string ref_path_filename = file_location + "/ref_path";

  saveTrajectoriesToJson(samples_, trajectories_filename + ".json");
  savePathToJson(path, ref_path_filename + ".json");

  std::string command =
      "python3 " + file_location + "/trajectory_sampler_plt.py --samples \"" +
      trajectories_filename + "\" --reference \"" + ref_path_filename + "\"";

  // Execute the Python script
  int res = system(command.c_str());
  if (res != 0)
      throw std::system_error(res, std::generic_category(),
                              "Python script failed with error code");

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
             "Goal not reached in " << counter << " steps");
}
