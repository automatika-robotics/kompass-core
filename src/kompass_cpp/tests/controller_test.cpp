#include "controllers/dwa.h"
#include "controllers/pure_pursuit.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
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

  // Normalize yaw
  while (robotState.yaw > M_PI)
    robotState.yaw -= 2.0 * M_PI;
  while (robotState.yaw < -M_PI)
    robotState.yaw += 2.0 * M_PI;
}

// --- Path Generators ---
Path::Path createStraightPath() {
  std::vector<Path::Point> points;
  for (double x = 0.0; x <= 10.0; x += 0.5) {
    points.emplace_back(x, 0.0, 0.0);
  }
  return Path::Path(points);
}

Path::Path createUTurnPath() {
  std::vector<Path::Point> points;
  // First straight segment
  for (double x = 0.0; x <= 5.0; x += 0.5) {
    points.emplace_back(x, 0.0, 0.0);
  }
  // Semi-circle turn
  double radius = 5.5;
  double center_x = 5.0;
  double center_y = 2.5;
  for (double angle = -M_PI_2; angle <= M_PI_2; angle += 0.2) {
    points.emplace_back(center_x + radius * std::cos(angle),
                        center_y + radius * std::sin(angle), 0.0);
  }
  // Return straight segment
  for (double x = 5.0; x >= 0.0; x -= 0.5) {
    points.emplace_back(x, 5.0, 0.0);
  }
  return Path::Path(points);
}

Path::Path createCirclePath() {
  // 3/4 of a circle
  std::vector<Path::Point> points;
  double radius = 10.0;
  for (double angle = 0.0; angle <= 3.0 * M_PI / 2.0; angle += 0.1) {
    points.emplace_back(radius * std::cos(angle), radius * std::sin(angle),
                        0.0);
  }
  return Path::Path(points);
}

// Creates a circular obstacle as a point cloud
std::vector<Path::Point> createRoundObstacle(double x, double y, double radius,
                                             double resolution = 0.1) {
  std::vector<Path::Point> cloud;
  for (double r = 0; r <= radius; r += resolution) {
    for (double theta = 0; theta < 2 * M_PI;
         theta +=
         resolution /
         r) { // Scale theta step by radius to keep density somewhat constant
      cloud.emplace_back(x + r * std::cos(theta), y + r * std::sin(theta), 0.0);
    }
    // Handle center point r=0 separately to avoid division by zero if loop
    // condition allows
    if (r == 0)
      cloud.emplace_back(x, y, 0.0);
  }
  return cloud;
}

// --- Test Case ---
BOOST_AUTO_TEST_CASE(test_PurePursuit_All_Scenarios) {
  double timeStep = 0.1;
  int max_steps = 1000;

  // Robot Limits
  Control::LinearVelocityControlParams x_params(1.0, 2.0, 2.0); // Max v=1.0 m/s
  Control::LinearVelocityControlParams y_params(1.0, 2.0, 2.0); // For Omni
  Control::AngularVelocityControlParams angular_params(0.7, 1.0, 2.0,
                                                       2.0); // Max omega
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);

  // config
  Control::PurePursuit::PurePursuitConfig pp_config;
  pp_config.setParameter("wheel_base", 0.34);
  pp_config.setParameter("speed_regulation_curvature",
                         0.5); // Speed regulation parameters
  pp_config.setParameter("speed_regulation_angular", 0.5);
  pp_config.setParameter("max_point_interpolation_distance", 0.05);
  pp_config.setParameter("path_segment_length", 1.0);
  pp_config.setParameter("goal_dist_tolerance", 0.3);

  // Define Scenarios
  std::map<std::string, Control::ControlType> robotTypes = {
      {"Ackermann", Control::ControlType::ACKERMANN},
      {"DiffDrive", Control::ControlType::DIFFERENTIAL_DRIVE},
      {"Omni", Control::ControlType::OMNI}
    };

  std::map<std::string, std::function<Path::Path()>> pathGenerators = {
      {"Straight", createStraightPath},
      {"UTurn", createUTurnPath},
      {"Circle", createCirclePath}
    };

  // Define Obstacle Locations for each path (approximate middle)
  std::map<std::string, std::pair<double, double>> obstacleLocations = {
      {"Straight", {4.0, 0.0}}, // Blocking the straight line
      {"UTurn",
       {10.0, 0.0}}, // Blocking the apex of the U-turn (approx x=7.5, y=2.5)
      {"Circle", {5.0, 8.5}} // Blocking the circle at x=4, y=0 (start/end area)
                             // - let's move it to x=0, y=4 (90 deg)
  };

  auto robotShapeType = Kompass::CollisionChecker::ShapeType::CYLINDER;
  std::vector<float> robotDimensions{0.1, 0.4};
  // std::array<float, 3> sensorPositionWRTbody {0.0, 0.0, 1.0};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 0.0};
  const Eigen::Vector4f sensor_rotation_body{0, 0, 0, 1};

  // Path for saving files
  boost::filesystem::path executablePath = boost::dll::program_location();
  std::string file_location = executablePath.parent_path().string();

  // We run two passes: one without obstacles (Standard) and one with
  // (Avoidance)
  std::vector<bool> useAvoidanceModes = {false, true};

  // Iterate through all combinations
  for (bool useAvoidance : useAvoidanceModes) {
    for (auto const &[robotName, robotType] : robotTypes) {
      for (auto const &[pathName, pathGen] : pathGenerators) {

        LOG_INFO("------------------------------------------------");
        LOG_INFO("Testing Pure Pursuit: Robot=", robotName,
                 ", Path=", pathName, ", Obstacle Avoidance=",
                 (useAvoidance ? "ON" : "OFF"));

        Control::PurePursuit controller(
            robotType, controlLimits, robotShapeType, robotDimensions,
            sensor_position_body, sensor_rotation_body, 0.1, pp_config);

        // Generate Path and Set to Controller
        Path::Path path = pathGen();
        controller.setCurrentPath(path);

        // Initialize Robot State
        Path::Point startPoint = path.getStart();
        float start_yaw = path.getStartOrientation();
        Path::State robotState(startPoint.x(), startPoint.y(), start_yaw, 0.0);

        // Slight offset to test convergence
        if (pathName == "Circle") {
          robotState.x += 0.2;
        }

        // Generate Obstacle (if avoidance mode)
        std::vector<Path::Point> obstacleCloud;
        if (useAvoidance) {
          auto loc = obstacleLocations[pathName];
          // Create a round obstacle of radius 0.3m
          obstacleCloud = createRoundObstacle(loc.first, loc.second, 0.3);
        }

        // Recorder
        Control::TrajectorySamples2D driven_trajectory(1, max_steps);
        Control::TrajectoryPath driven_path(max_steps);
        Control::TrajectoryVelocities2D driven_vels(max_steps);

        int step = 0;
        bool goal_reached = false;
        Control::Controller::Result result;
        // Simulation Loop
        while (!goal_reached && step < max_steps) {
          // Execute Controller
          if (useAvoidance) {
            result = controller.execute(robotState, timeStep, obstacleCloud);
          } else {
            result = controller.execute(robotState, timeStep);
          }

          // Record
          driven_path.add(step, robotState.x, robotState.y, 0.0);
          if (step < max_steps - 1) {
            driven_vels.add(step, result.velocity_command);
          }
          if (result.status ==
              Control::Controller::Result::Status::GOAL_REACHED) {
            LOG_INFO(FGRN("Goal Reached at step "), step);
            goal_reached = true;
            driven_path.numPointsPerTrajectory_ = step + 1;
            driven_vels.numPointsPerTrajectory_ = step + 1;
          } else if (result.status ==
                     Control::Controller::Result::Status::NO_COMMAND_POSSIBLE) {
            LOG_ERROR(FRED("Controller failed to find command"));
            break;
          }

          // Apply control
          controller.setCurrentVelocity(result.velocity_command);
          applyControl(robotState, result.velocity_command, timeStep);
          step++;
        }

        if (!goal_reached) {
          LOG_WARNING(
              FYEL("Max steps reached without hitting goal tolerance."));
          driven_path.numPointsPerTrajectory_ = step;
        }

        Control::TrajectorySamples2D sample_container(1, step);

        // Construct Trajectories for Plotting
        Control::TrajectoryPath final_path_trace(step);
        Control::TrajectoryVelocities2D final_vel_trace(step);
        for (int i = 0; i < step; ++i) {
          final_path_trace.add(i, driven_path.x(i), driven_path.y(i));
          if (i < step - 1) {
            final_vel_trace.add(i, driven_vels.getIndex(i));
          }
        }

        sample_container.push_back(final_vel_trace, final_path_trace);

        std::string filename_base =
            "pure_pursuit_" + robotName + "_" + pathName;
        std::string traj_file = file_location + "/" + filename_base + "_traj";
        std::string ref_file = file_location + "/" + filename_base + "_ref";
        std::string obs_file = file_location + "/" + filename_base + "_obs";

        saveTrajectoriesToJson(sample_container, traj_file + ".json");
        savePathToJson(path, ref_file + ".json");

        // Plot using the python script
        std::string command;
        // Save obstacles if present
        if (useAvoidance) {
          // Create a dummy path container for obstacles to reuse
          // savePathToJson/plotter or we create a specific function. Let's try
          // to reuse savePathToJson by wrapping points.
          Path::Path obsPathObj(obstacleCloud); // +1 buffer
          savePathToJson(obsPathObj, obs_file + ".json");
          // Plot using the python script
          command = "python3 " + file_location +
                    "/trajectory_sampler_plt.py --samples \"" + traj_file +
                    "\" --reference \"" + ref_file + "\" --obstacles \"" +
                    obs_file + "\"";
        } else {
          // Plot using the python script
          command = "python3 " + file_location +
                    "/trajectory_sampler_plt.py --samples \"" + traj_file +
                    "\" --reference \"" + ref_file + "\"";
        }

        int res = system(command.c_str());
        if (res != 0)
          throw std::system_error(res, std::generic_category(),
                                  "Python script failed with error code");
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_DWA) {
  // Create timer
  Timer time;

  // Create a test path
  std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0), Path::Point(1.0, 0.0, 0.0),
                                  Path::Point(2.0, 0.0, 0.0)};
  Path::Path path(points);

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

  while (!planner.isGoalReached() and counter < 150) {
    counter++;
    // Set the robot state in the planner
    planner.setCurrentState(robotState);

    Control::TrajSearchResult result =
        planner.computeVelocityCommandsSet(robotControl, robotScan);

    LOG_INFO(FMAG("Result Found: "), result.isTrajFound);

    auto current_path = planner.getCurrentPath();

    LOG_DEBUG(FMAG("Current segment: "), planner.getCurrentSegmentIndex(),
              FMAG(", Max segments: "), current_path.getNumSegments());

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
