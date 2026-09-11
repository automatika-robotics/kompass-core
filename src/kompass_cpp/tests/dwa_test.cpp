#include "controllers/dwa.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <system_error>
#define BOOST_TEST_MODULE KOMPASS DWA TESTS
#include "controller_test_helpers.h"
#include "json_export.h"
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

using namespace Kompass;

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
  costWeights.setParameter("goal_distance_weight", 1.0);
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
  Control::LaserScan robotScan(toVecF({0.4, 0.3}), toVecF({10, 10.1}));

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

// A goal closer than the smallest grid speed covers within the horizon: 0.5 s
// steps, a 5 s rollout and a speed grid whose smallest speed (0.2 m/s)
// travels 0.9 m, twice the distance to the goal. Every straight grid sample
// overshoots, so only the stop sample and the creep speed can end near the
// goal, and with the horizon capped at the goal the grid speeds can end on
// it too. The controller must reach it with forward motion only: no
// reverse, turning, or overshoot, and within a few steps rather than by
// creeping the whole way.
BOOST_AUTO_TEST_CASE(test_DWA_creeps_onto_a_close_goal) {
  std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0),
                                  Path::Point(1.5, 0.0, 0.0),
                                  Path::Point(3.0, 0.0, 0.0)};
  Path::Path path(points);

  const double timeStep = 0.5;
  const double predictionHorizon = 5.0;
  const double controlHorizon = 2.5;
  // 9 speeds over [-0.8, 0.8]: +/-0.2, +/-0.4, +/-0.6, +/-0.8 and a zero
  // the sampler skips, exactly the grid seen in the bag
  const int maxLinearSamples = 9;
  const int maxAngularSamples = 10;

  Control::CostEvaluator::TrajectoryCostsWeights costWeights;
  costWeights.setParameter("reference_path_distance_weight", 1.0);
  costWeights.setParameter("goal_distance_weight", 1.0);
  costWeights.setParameter("obstacles_distance_weight", 0.0);
  costWeights.setParameter("smoothness_weight", 0.0);
  costWeights.setParameter("jerk_weight", 0.0);

  // The creep speed is the minimum speed of the x-axis limits
  Control::LinearVelocityControlParams x_params(0.8, 5.0, 10.0, 0.05);
  Control::LinearVelocityControlParams y_params(0.0, 0.0, 0.0, 0.0);
  Control::AngularVelocityControlParams angular_params(M_PI, 1.5, 3.0, 3.0);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);
  std::vector<float> robotDimensions{0.2, 0.4};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 0.0};
  const Eigen::Vector4f sensor_rotation_body{0, 0, 0, 1};

  Control::DWA planner(controlLimits, Control::ControlType::DIFFERENTIAL_DRIVE,
                       timeStep, predictionHorizon, controlHorizon,
                       maxLinearSamples, maxAngularSamples,
                       Kompass::CollisionChecker::ShapeType::CYLINDER,
                       robotDimensions, sensor_position_body,
                       sensor_rotation_body, 0.1, costWeights, 1);
  planner.setCurrentPath(path);

  // Nothing within 20 m in any direction
  Eigen::VectorXf ranges = Eigen::VectorXf::Constant(36, 20.0f);
  Eigen::VectorXf angles(36);
  for (int i = 0; i < 36; ++i) {
    angles(i) = static_cast<float>(i) * 2.0f * M_PI / 36;
  }
  Control::LaserScan robotScan(ranges, angles);

  // 0.45 m short of the goal, facing it, at rest
  Path::State robotState(2.55, 0.0, 0.0, 0.0);
  Control::Velocity2D robotControl;
  const double start_distance = 0.45;

  int counter = 0;
  while (!planner.isGoalReached() and counter < 60) {
    counter++;
    planner.setCurrentState(robotState);
    Control::TrajSearchResult result =
        planner.computeVelocityCommandsSet(robotControl, robotScan);
    BOOST_REQUIRE_MESSAGE(result.isTrajFound,
                          "No command found at step " << counter);

    const double vx = planner.getLinearVelocityCmdX();
    const double omega = planner.getAngularVelocityCmd();
    BOOST_TEST_CONTEXT("step " << counter << ", x " << robotState.x) {
      BOOST_TEST(vx >= 0.0);
      BOOST_TEST(std::abs(omega) < 0.3);
    }
    // The chosen command becomes the measured velocity of the next step
    robotControl = Control::Velocity2D(vx, 0.0, omega);
    applyControl(robotState, robotControl, timeStep);
    const double distance = std::hypot(3.0 - robotState.x, robotState.y);
    BOOST_TEST(distance <= start_distance + 0.05);
  }
  BOOST_TEST(planner.isGoalReached(),
             "Goal not reached in " << counter << " steps");
  // Four steps with the goal-aware horizon; fifteen when creeping at the
  // minimum speed the whole way
  BOOST_TEST(counter <= 6);
}

// The rollout horizon follows the distance left to the goal: the base
// horizon far away, the time to reach the goal at full speed (rounded to
// steps, never below two steps) close to it.
BOOST_AUTO_TEST_CASE(test_DWA_horizon_caps_at_the_goal) {
  // Reads the sampler's rollout length and the tracked remaining distance,
  // which the planner keeps protected
  struct Probe : Control::DWA {
    using Control::DWA::DWA;
    size_t rolloutPoints() const { return trajSampler->numPointsPerTrajectory; }
    double remainingDistance() const {
      return currentPath->totalPathLength() -
             currentPath->getDistanceAtIndex(closestPosition->index);
    }
  };

  std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0),
                                  Path::Point(5.0, 0.0, 0.0),
                                  Path::Point(10.0, 0.0, 0.0)};
  Path::Path path(points);

  const double timeStep = 0.5;
  const double predictionHorizon = 5.0;
  Control::CostEvaluator::TrajectoryCostsWeights costWeights;
  costWeights.setParameter("reference_path_distance_weight", 1.0);
  costWeights.setParameter("goal_distance_weight", 1.0);
  costWeights.setParameter("obstacles_distance_weight", 0.0);
  costWeights.setParameter("smoothness_weight", 0.0);
  costWeights.setParameter("jerk_weight", 0.0);
  Control::LinearVelocityControlParams x_params(0.8, 5.0, 10.0, 0.05);
  Control::LinearVelocityControlParams y_params(0.0, 0.0, 0.0, 0.0);
  Control::AngularVelocityControlParams angular_params(M_PI, 1.5, 3.0, 3.0);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);
  Probe planner(controlLimits, Control::ControlType::DIFFERENTIAL_DRIVE,
                timeStep, predictionHorizon, 2.5, 9, 10,
                Kompass::CollisionChecker::ShapeType::CYLINDER,
                std::vector<float>{0.2, 0.4}, Eigen::Vector3f{0.0, 0.0, 0.0},
                Eigen::Vector4f{0, 0, 0, 1}, 0.1, costWeights, 1);
  planner.setCurrentPath(path);

  Eigen::VectorXf ranges = Eigen::VectorXf::Constant(36, 20.0f);
  Eigen::VectorXf angles(36);
  for (int i = 0; i < 36; ++i) {
    angles(i) = static_cast<float>(i) * 2.0f * M_PI / 36;
  }
  Control::LaserScan robotScan(ranges, angles);

  // The goal tolerance of the follower defaults to 0.1 m
  const double goal_tolerance = 0.1;
  const double v_max = 0.8;
  auto expected_points = [&](double remaining) {
    const double horizon = (remaining + goal_tolerance) / v_max + timeStep;
    return Control::getNumPointsPerTrajectory(
        timeStep, std::clamp(horizon, 3.0 * timeStep, predictionHorizon));
  };

  // Moves the robot along the path in small steps with a planning tick at
  // each, as the follower's segment tracking follows the state tick by tick
  // and cannot jump across segments
  double x = 0.0;
  auto driveTo = [&](double target_x) {
    while (x < target_x - 1e-9) {
      x = std::min(x + 0.25, target_x);
      planner.setCurrentState(Path::State(x, 0.0, 0.0, 0.0));
      planner.computeVelocityCommandsSet(Control::Velocity2D(), robotScan);
    }
  };

  // 9.5 m from the goal: the base horizon
  driveTo(0.5);
  BOOST_TEST(planner.rolloutPoints() ==
             Control::getNumPointsPerTrajectory(timeStep, predictionHorizon));

  // About 2 m from the goal: the remaining path plus the tolerance at
  // 0.8 m/s, plus one step, rounded down to whole steps
  driveTo(8.0);
  BOOST_TEST(planner.remainingDistance() > 1.5);
  BOOST_TEST(planner.remainingDistance() < 2.5);
  BOOST_TEST(planner.rolloutPoints() ==
             expected_points(planner.remainingDistance()));
  BOOST_TEST(planner.rolloutPoints() <
             Control::getNumPointsPerTrajectory(timeStep, predictionHorizon));

  // 0.45 m from the goal: the two-step floor
  driveTo(9.55);
  BOOST_TEST(planner.rolloutPoints() ==
             expected_points(planner.remainingDistance()));
  BOOST_TEST(planner.rolloutPoints() == 3u);
}

// DWA scenario matrix: same {robot × path × avoidance} cross-product the
// PurePursuit test uses, with the same obstacle locations. Verifies DWA can
// drive each robot type along each reference path, with and without a
// blocking obstacle, without colliding and without timing out.
BOOST_AUTO_TEST_CASE(test_DWA_All_Scenarios) {
  const double timeStep = 0.1;
  const double predictionHorizon = 4.0;
  const double controlHorizon = 0.5;
  const int maxLinearSamples = 20;
  const int maxAngularSamples = 20;
  const int maxNumThreads = 10;
  const double octreeRes = 0.1;

  const int max_steps = 1000;

  // Robot limits (same v/omega ceilings across all three robot types)
  Control::LinearVelocityControlParams x_params(1.0, 2.0, 2.0);
  Control::LinearVelocityControlParams y_params(1.0, 2.0, 2.0);
  Control::AngularVelocityControlParams angular_params(2.0, 2.0, 3.0, 3.0);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);

  // Cost weights
  Control::CostEvaluator::TrajectoryCostsWeights costWeights;
  costWeights.setParameter("reference_path_distance_weight", 1.0);
  costWeights.setParameter("goal_distance_weight", 1.0);
  costWeights.setParameter("obstacles_distance_weight", 0.0);
  costWeights.setParameter("smoothness_weight", 0.0);
  costWeights.setParameter("jerk_weight", 0.0);

  const std::map<std::string, Control::ControlType> robotTypes = {
      {"Ackermann", Control::ControlType::ACKERMANN},
      {"DiffDrive", Control::ControlType::DIFFERENTIAL_DRIVE},
      {"Omni", Control::ControlType::OMNI}};

  const std::map<std::string, std::function<Path::Path()>> pathGenerators = {
      {"Straight", createStraightPath},
      {"UTurn", createUTurnPath},
      {"Circle", createCirclePath}};

  // Obstacle placed on each path to force a detour.
  // Same obstacle locations as the PurePursuit matrix.
  const std::map<std::string, std::pair<double, double>> obstacleLocations = {
      {"Straight", {4.0, 0.0}},
      {"UTurn", {10.0, 0.0}},
      {"Circle", {5.0, 8.5}}
    };

  const auto robotShapeType = Kompass::CollisionChecker::ShapeType::CYLINDER;
  const float robotRadius = 0.1f;
  const std::vector<float> robotDimensions{robotRadius, 0.4f};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 0.0};
  const Eigen::Vector4f sensor_rotation_body{0, 0, 0, 1};

  boost::filesystem::path executablePath = boost::dll::program_location();
  std::string file_location = executablePath.parent_path().string();

  const std::vector<bool> useAvoidanceModes = {false, true};

  for (bool useAvoidance : useAvoidanceModes) {
    for (auto const &[robotName, robotType] : robotTypes) {
      for (auto const &[pathName, pathGen] : pathGenerators) {
        LOG_INFO("------------------------------------------------");
        LOG_INFO("Testing DWA: Robot=", robotName, ", Path=", pathName,
                 ", Obstacle Avoidance=", (useAvoidance ? "ON" : "OFF"));

        Control::DWA planner(controlLimits, robotType, timeStep,
                             predictionHorizon, controlHorizon,
                             maxLinearSamples, maxAngularSamples, robotShapeType,
                             robotDimensions, sensor_position_body,
                             sensor_rotation_body, octreeRes, costWeights,
                             maxNumThreads);

        // Match PurePursuit's goal tolerance — DWA's default 0.1 m is tight
        // given sampling resolution and sidestep-then-continue behavior.
        Control::Follower::FollowerParameters fp;
        fp.setParameter("goal_dist_tolerance", 0.3);
        planner.setParams(fp);

        Path::Path path = pathGen();
        planner.setCurrentPath(path);

        Path::Point startPoint = path.getStart();
        float start_yaw = path.getStartOrientation();
        Path::State robotState(startPoint.x(), startPoint.y(), start_yaw, 0.0);

        // Offset the circle start slightly so the robot has to converge onto
        // the path — same trick the PurePursuit matrix uses.
        if (pathName == "Circle") {
          robotState.x += 0.2;
        }

        std::vector<Path::Point> obstacleCloud;
        if (useAvoidance) {
          auto loc = obstacleLocations.at(pathName);
          obstacleCloud = createRoundObstacle(loc.first, loc.second, 0.3);
        }

        Control::TrajectoryPath driven_path(max_steps);
        Control::TrajectoryVelocities2D driven_vels(max_steps);
        driven_path.x.setZero();
        driven_path.y.setZero();
        driven_path.z.setZero();
        driven_vels.vx.setZero();
        driven_vels.vy.setZero();
        driven_vels.omega.setZero();

        Control::Velocity2D robotControl;
        double min_clearance = std::numeric_limits<double>::infinity();
        int step = 0;
        bool goal_reached = false;
        bool planner_failed = false;

        while (!goal_reached && step < max_steps) {
          driven_path.add(step, robotState.x, robotState.y, 0.0);
          planner.setCurrentState(robotState);

          Control::TrajSearchResult result =
              planner.computeVelocityCommandsSet(robotControl, obstacleCloud);

          if (!result.isTrajFound) {
            LOG_ERROR(FRED("DWA no feasible trajectory at step "), step);
            planner_failed = true;
            break;
          }

          Control::Velocity2D cmd(planner.getLinearVelocityCmdX(),
                                  planner.getLinearVelocityCmdY(),
                                  planner.getAngularVelocityCmd());
          driven_vels.add(step, cmd);
          robotControl = cmd;
          applyControl(robotState, cmd, timeStep);

          if (useAvoidance) {
            double clearance = minDistanceToCloud(robotState.x, robotState.y,
                                                  obstacleCloud);
            if (clearance < min_clearance)
              min_clearance = clearance;
          }

          if (planner.isGoalReached()) {
            LOG_INFO(FGRN("Goal reached at step "), step);
            goal_reached = true;
          }
          step++;
        }

        // Plot
        const int traced_steps = std::max(step, 1);
        Control::TrajectorySamples2D samples(1, traced_steps);
        Control::TrajectoryPath final_path(traced_steps);
        Control::TrajectoryVelocities2D final_vels(traced_steps);
        for (int i = 0; i < traced_steps; ++i) {
          final_path.add(i, driven_path.x(i), driven_path.y(i));
          if (i < traced_steps - 1) {
            final_vels.add(i, driven_vels.getIndex(i));
          }
        }
        samples.push_back(final_vels, final_path);

        std::string filename_base = "dwa_" + robotName + "_" + pathName +
                                    (useAvoidance ? "_avoid" : "");
        std::string traj_file = file_location + "/" + filename_base + "_traj";
        std::string ref_file = file_location + "/" + filename_base + "_ref";
        std::string obs_file = file_location + "/" + filename_base + "_obs";

        saveTrajectoriesToJson(samples, traj_file + ".json");
        savePathToJson(path, ref_file + ".json");

        std::string command;
        if (useAvoidance) {
          Path::Path obsPathObj(obstacleCloud);
          savePathToJson(obsPathObj, obs_file + ".json");
          command = "python3 " + file_location +
                    "/trajectory_sampler_plt.py --samples \"" + traj_file +
                    "\" --reference \"" + ref_file + "\" --obstacles \"" +
                    obs_file + "\"";
        } else {
          command = "python3 " + file_location +
                    "/trajectory_sampler_plt.py --samples \"" + traj_file +
                    "\" --reference \"" + ref_file + "\"";
        }
        int res = system(command.c_str());
        if (res != 0)
          throw std::system_error(res, std::generic_category(),
                                  "Python script failed with error code");

        LOG_INFO("DWA ", robotName, "/", pathName, "/",
                 (useAvoidance ? "ON" : "OFF"),
                 ": goal_reached=", goal_reached, ", steps=", step,
                 ", planner_failed=", planner_failed,
                 ", min_clearance=", min_clearance);

        const std::string tag =
            robotName + "/" + pathName + "/" + (useAvoidance ? "ON" : "OFF");
        BOOST_TEST(!planner_failed,
                   "DWA failed to find trajectory: " << tag);
        BOOST_TEST(goal_reached, "DWA did not reach goal: " << tag);
        if (useAvoidance) {
          BOOST_TEST(min_clearance >= robotRadius,
                     "DWA collided with obstacle: " << tag);
        }
      }
    }
  }
}
