#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <memory>
#define BOOST_TEST_MODULE KOMPASS TESTS
#include "json_export.h"
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass;

struct VisionDWATestConfig {
  // Sampling configuration
  double timeStep;
  double predictionHorizon;
  double controlHorizon;
  int maxLinearSamples;
  int maxAngularSamples;
  int maxNumThreads;

  // Octomap resolution
  double octreeRes;

  // Cost weights
  Control::CostEvaluator::TrajectoryCostsWeights costWeights;

  // Robot configuration
  Control::LinearVelocityControlParams x_params;
  Control::LinearVelocityControlParams y_params;
  Control::AngularVelocityControlParams angular_params;
  Control::ControlLimitsParams controlLimits;
  Control::ControlType controlType;
  Kompass::CollisionChecker::ShapeType robotShapeType;
  std::vector<float> robotDimensions;
  const std::array<float, 3> sensorPositionWRTbody;
  const std::array<float, 4> sensorRotationWRTbody;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud;

  // Robot start state (pose)
  Path::State robotState;

  // Tracked target with respect to the robot
  Control::Velocity2D tracked_vel;
  Control::TrackedPose2D tracked_pose;
  // VisionDWA configuration object
  std::unique_ptr<Control::VisionDWA> controller;

  // Constructor to initialize the struct
  VisionDWATestConfig(const float timeStep, const float predictionHorizon,
                      const float controlHorizon, const int maxLinearSamples,
                      const int maxAngularSamples,
                      const std::vector<Path::Point> sensor_points,
                      const float maxVel = 1.0, const float maxOmega = 2.0,
                      const int maxNumThreads = 1,
                      const double reference_path_distance_weight = 1.0,
                      const double goal_distance_weight = 0.0,
                      const double obstacles_distance_weight = 0.5)
      : timeStep(timeStep), predictionHorizon(predictionHorizon),
        controlHorizon(controlHorizon), maxLinearSamples(maxLinearSamples),
        maxAngularSamples(maxAngularSamples), maxNumThreads(maxNumThreads),
        octreeRes(0.1), x_params(maxVel, 5.0, 10.0), y_params(1, 3, 5),
        angular_params(3.14, maxOmega, 3.0, 3.0),
        controlLimits(x_params, y_params, angular_params),
        controlType(Control::ControlType::ACKERMANN),
        robotShapeType(Kompass::CollisionChecker::ShapeType::CYLINDER),
        robotDimensions{0.1, 0.4}, sensorPositionWRTbody{0.0, 0.0, 0.0},
        sensorRotationWRTbody{0, 0, 0, 1}, cloud(sensor_points),
        robotState(-0.5, 0.0, 0.0, 0.0), tracked_vel(0.1, 0.0, 0.3),
        tracked_pose(0.0, 0.0, 0.0, tracked_vel) {
    // Initialize cost weights
    costWeights.setParameter("reference_path_distance_weight",
                             reference_path_distance_weight);
    costWeights.setParameter("goal_distance_weight", goal_distance_weight);
    costWeights.setParameter("obstacles_distance_weight",
                             obstacles_distance_weight);
    costWeights.setParameter("smoothness_weight", 1.0);
    costWeights.setParameter("jerk_weight", 0.0);
    controller = std::make_unique<Control::VisionDWA>(
        controlType, controlLimits, maxLinearSamples, maxAngularSamples,
        robotShapeType, robotDimensions, sensorPositionWRTbody,
        sensorRotationWRTbody, octreeRes, costWeights);
  }

  bool run_test(const int numPointsPerTrajectory, std::string pltFileName) {
    Control::TrajectorySamples2D samples(2, numPointsPerTrajectory);
    Control::TrajectoryVelocities2D simulated_velocities(
        numPointsPerTrajectory);
    Control::TrajectoryPath robot_path(numPointsPerTrajectory),
        tracked_path(numPointsPerTrajectory);
    Control::Velocity2D cmd;

    int step = 0;
    while (step < numPointsPerTrajectory) {
      Path::Point point(robotState.x, robotState.y, 0.0);
      robot_path.add(step, point);
      tracked_path.add(step, {tracked_pose.x(), tracked_pose.y(), 0.0});
      controller->setCurrentState(robotState);

      Control::TrajSearchResult result =
          controller->getTrackingCtrl(tracked_pose, cmd, cloud);
      // cmd = controller.getPureTrackingCtrl(tracked_pose);
      // robotState.update(cmd, timeStep);

      if (result.isTrajFound) {
        LOG_INFO(FMAG("STEP: "), step,
                 FMAG(", Found best trajectory with cost: "), result.trajCost);
        if (controller->getLinearVelocityCmdX() > controlLimits.velXParams.maxVel) {
          LOG_ERROR(BOLD(FRED("Vx is larger than max vel: ")),
                    KRED, controller->getLinearVelocityCmdX(), RST,
                    BOLD(FRED(", Vx: ")), KRED, controlLimits.velXParams.maxVel,
                    RST);
          return false;
        }
        cmd.setVx(controller->getLinearVelocityCmdX());
        cmd.setVy(controller->getLinearVelocityCmdY());
        cmd.setOmega(controller->getAngularVelocityCmd());

        LOG_DEBUG(BOLD(FBLU("Robot at: {x: ")), KBLU, robotState.x, RST,
                  BOLD(FBLU(", y: ")), KBLU, robotState.y, RST,
                  BOLD(FBLU(", yaw: ")), KBLU, robotState.yaw, RST,
                  BOLD(FBLU("}")));

        LOG_DEBUG(BOLD(FGRN("Found Control: {Vx: ")), KGRN, cmd.vx(), RST,
                  BOLD(FGRN(", Vy: ")), KGRN, cmd.vy(), RST,
                  BOLD(FGRN(", Omega: ")), KGRN, cmd.omega(), RST,
                  BOLD(FGRN("}")));

        robotState.update(cmd, timeStep);
      } else {
        LOG_ERROR(BOLD(FRED("DWA Planner failed with robot at: {x: ")), KRED,
                  robotState.x, RST, BOLD(FRED(", y: ")), KRED, robotState.y,
                  RST, BOLD(FRED(", yaw: ")), KRED, robotState.yaw, RST,
                  BOLD(FRED("}")));
        return false;
      }

      tracked_pose.update(timeStep);
      step++;
    }
    samples.push_back(simulated_velocities, robot_path);
    samples.push_back(simulated_velocities, tracked_path);


    // Plot the trajectories (Save to json then run python script for plotting)
    boost::filesystem::path executablePath = boost::dll::program_location();
    std::string file_location = executablePath.parent_path().string();

    std::string trajectories_filename = file_location + "/" + pltFileName;

    saveTrajectoriesToJson(samples, trajectories_filename + ".json");
    // savePathToJson(reference_segment, ref_path_filename + ".json");

    std::string command = "python3 " + file_location +
                          "/trajectory_sampler_plt.py --samples \"" +
                          trajectories_filename + "\"";

    // Execute the Python script
    int res = system(command.c_str());
    if (res != 0)
      throw std::system_error(res, std::generic_category(),
                              "Python script failed with error code");
    return true;
  }
};

BOOST_AUTO_TEST_CASE(test_VisionDWA_obstacle_free) {
  // Create timer
  Timer time;

  // Sampling configuration
  double timeStep = 0.1;
  double predictionHorizon =0.5;
  double controlHorizon = 0.2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};

  VisionDWATestConfig testConfig(timeStep, predictionHorizon, controlHorizon, maxLinearSamples, maxAngularSamples, cloud);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(numPointsPerTrajectory, std::string("vision_follower_obstacle_free"));
  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
}

BOOST_AUTO_TEST_CASE(test_VisionDWA_with_obstacle) {
  // Create timer
  Timer time;

  // Sampling configuration
  double timeStep = 0.1;
  double predictionHorizon = 0.5;
  double controlHorizon = 0.2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{0.3, 0.27, 0.1}};

  VisionDWATestConfig testConfig(timeStep, predictionHorizon, controlHorizon,
                                 maxLinearSamples, maxAngularSamples, cloud);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory, std::string("vision_follower_with_obstacle"));
  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
}
