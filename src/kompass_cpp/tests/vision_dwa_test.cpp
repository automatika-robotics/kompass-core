#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
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
  int predictionHorizon;
  int controlHorizon;
  int maxLinearSamples;
  int maxAngularSamples;
  int maxNumThreads;

  // Detected Boxes
  std::vector<Bbox3D> detected_boxes;
  int num_test_boxes = 3;
  Eigen::Vector2i ref_point_img{150, 150};
  float boxes_ori = 0.0f;

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
  VisionDWATestConfig(const double timeStep, const int predictionHorizon,
                      const int controlHorizon, const int maxLinearSamples,
                      const int maxAngularSamples,
                      const std::vector<Path::Point> sensor_points,
                      const bool use_tracker = true,
                      const float maxVel = 1.0, const float maxOmega = 4.0,
                      const int maxNumThreads = 1,
                      const double reference_path_distance_weight = 5.0,
                      const double goal_distance_weight = 1.0,
                      const double obstacles_distance_weight = 0.1)
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
    costWeights.setParameter("smoothness_weight", 0.0);
    costWeights.setParameter("jerk_weight", 0.0);
    auto config = Control::VisionDWA::VisionDWAConfig();
    config.setParameter("control_time_step", timeStep);
    config.setParameter("control_horizon", controlHorizon);
    config.setParameter("prediction_horizon", predictionHorizon);
    controller = std::make_unique<Control::VisionDWA>(
        controlType, controlLimits, maxLinearSamples, maxAngularSamples,
        robotShapeType, robotDimensions, sensorPositionWRTbody,
        sensorRotationWRTbody, octreeRes, costWeights, maxNumThreads, config,
        use_tracker);

    // Initialize the detected boxes
    Bbox3D new_box;
    new_box.size = {0.5f, 0.5f, 1.0f};
    detected_boxes.resize(num_test_boxes - 1);
    for (int i = 0; i < num_test_boxes - 1; ++i) {
      auto new_box_shift =
          Eigen::Vector3f({float(0.7 * i), float(0.7 * i), 0.0f});
      auto img_frame_shift =
          Eigen::Vector2i({float(50 * i), float(50 * i)});
      new_box.center = new_box_shift;
      new_box.center_img_frame = img_frame_shift + ref_point_img;
      new_box.size_img_frame = {25, 25};
      detected_boxes[i] = new_box;
    }
  }

  void moveDetectedBoxes() {
    // Update the detected boxes using the velocity command
    Eigen::Vector3f target_ref_vel = {float(tracked_vel.vx() * cos(boxes_ori)),
                                      float(tracked_vel.vx() * sin(boxes_ori)),
                                      0.0f};
    boxes_ori += tracked_vel.omega() * timeStep;
    for (auto &box : detected_boxes) {
      box.center += target_ref_vel * timeStep;
    }
  };

  bool run_test(const int numPointsPerTrajectory, std::string pltFileName, bool with_tracker) {
    Control::TrajectorySamples2D samples(2, numPointsPerTrajectory);
    Control::TrajectoryVelocities2D simulated_velocities(
        numPointsPerTrajectory);
    Control::TrajectoryPath robot_path(numPointsPerTrajectory),
        tracked_path(numPointsPerTrajectory);
    Control::Velocity2D cmd;
    Control::TrajSearchResult result;

    if(with_tracker){
      controller->setInitialTracking(
          ref_point_img(0), ref_point_img(1), detected_boxes);
    }

    int step = 0;
    while (step < numPointsPerTrajectory) {
      Path::Point point(robotState.x, robotState.y, 0.0);
      robot_path.add(step, point);
      tracked_path.add(step, {tracked_pose.x(), tracked_pose.y(), 0.0});
      controller->setCurrentState(robotState);

      if(with_tracker){
        result =
            controller->getTrackingCtrl(detected_boxes, cmd, cloud);
      }else{
        result = controller->getTrackingCtrl(tracked_pose, cmd, cloud);
      }

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
      if(with_tracker){
        moveDetectedBoxes();
      }
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
  int predictionHorizon = 10;
  int controlHorizon = 2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};

  VisionDWATestConfig testConfig(timeStep, predictionHorizon, controlHorizon, maxLinearSamples, maxAngularSamples, cloud, false);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory, std::string("vision_follower_obstacle_free"), false);
  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
}

BOOST_AUTO_TEST_CASE(test_VisionDWA_with_obstacle) {
  // Create timer
  Timer time;

  // Sampling configuration
  double timeStep = 0.1;
  int predictionHorizon = 10;
  int controlHorizon = 2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{0.3, 0.27, 0.1}};

  VisionDWATestConfig testConfig(timeStep, predictionHorizon, controlHorizon,
                                 maxLinearSamples, maxAngularSamples, cloud, false);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory, std::string("vision_follower_with_obstacle"), false);
  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
}

BOOST_AUTO_TEST_CASE(test_VisionDWA_with_tracker_obs_free) {
  // Create timer
  Timer time;

  // Sampling configuration
  double timeStep = 0.1;
  int predictionHorizon = 10;
  int controlHorizon = 2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};

  VisionDWATestConfig testConfig(timeStep, predictionHorizon, controlHorizon,
                                 maxLinearSamples, maxAngularSamples, cloud,
                                 true);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory, std::string("vision_follower_with_tracker"), true);
  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
}

BOOST_AUTO_TEST_CASE(test_VisionDWA_with_tracker_and_obstacle) {
  // Create timer
  Timer time;

  // Sampling configuration
  double timeStep = 0.1;
  int predictionHorizon = 10;
  int controlHorizon = 2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{0.3, 0.27, 0.1}};

  VisionDWATestConfig testConfig(timeStep, predictionHorizon, controlHorizon,
                                 maxLinearSamples, maxAngularSamples, cloud,
                                 true);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory, std::string("vision_follower_with_tracker_and_obstacle"), true);
  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
}
