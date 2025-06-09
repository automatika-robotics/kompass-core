#include "controllers/vision_dwa.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <memory>
#include <opencv2/opencv.hpp>
#define BOOST_TEST_MODULE KOMPASS TESTS
#include "json_export.h"
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass;

struct VisionDWATestConfig {
  bool use_local_frame;

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
  Eigen::Vector3f prox_sensor_position_body;
  Eigen::Vector4f prox_sensor_rotation_body;

  // Depth detector
  Eigen::Vector2f focal_length = {911.71, 910.288};
  Eigen::Vector2f principal_point = {643.06, 366.72};
  Eigen::Vector2f depth_range = {0.001, 5.0}; // 5cm to 5 meters
  float depth_conv_factor = 1e-3;             // convert from mm to m
  Eigen::Isometry3f camera_body_tf;

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
  VisionDWATestConfig(const std::vector<Path::Point> sensor_points,
                      const bool use_local_frame = true,
                      const double timeStep = 0.1,
                      const int predictionHorizon = 6,
                      const int controlHorizon = 2,
                      const int maxLinearSamples = 20,
                      const int maxAngularSamples = 20,
                      const float maxVel = 2.0, const float maxOmega = 4.0,
                      const int maxNumThreads = 1,
                      const double reference_path_distance_weight = 1.0,
                      const double goal_distance_weight = 1.0,
                      const double obstacles_distance_weight = 0.0)
      : use_local_frame(use_local_frame), timeStep(timeStep), predictionHorizon(predictionHorizon),
        controlHorizon(controlHorizon), maxLinearSamples(maxLinearSamples),
        maxAngularSamples(maxAngularSamples), maxNumThreads(maxNumThreads),
        octreeRes(0.1), x_params(maxVel, 5.0, 10.0), y_params(1, 3, 5),
        angular_params(3.14, maxOmega, 3.0, 3.0),
        controlLimits(x_params, y_params, angular_params),
        controlType(Control::ControlType::ACKERMANN),
        robotShapeType(Kompass::CollisionChecker::ShapeType::CYLINDER),
        robotDimensions{0.1, 0.4}, prox_sensor_position_body{0.0, 0.0, 0.0},
        prox_sensor_rotation_body{0, 0, 0, 1}, cloud(sensor_points),
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
    config.setParameter("use_local_coordinates", use_local_frame);

    // For depth config
    // Body to camera tf from robot of test pictures
    auto body_to_link_tf =
        getTransformation(Eigen::Quaternionf{0.0f, 0.1987f, 0.0f, 0.98f},
                          Eigen::Vector3f{0.32f, 0.0209f, 0.3f});

    auto link_to_cam_tf =
        getTransformation(Eigen::Quaternionf{0.01f, -0.00131f, 0.002f, 0.9999f},
                          Eigen::Vector3f{0.0f, 0.0105f, 0.0f});

    auto cam_to_cam_opt_tf =
        getTransformation(Eigen::Quaternionf{-0.5f, 0.5f, -0.5f, 0.5f},
                          Eigen::Vector3f{0.0f, 0.0105f, 0.0f});

    Eigen::Isometry3f body_to_cam_tf =
        body_to_link_tf * link_to_cam_tf * cam_to_cam_opt_tf;

    Eigen::Vector3f translation = body_to_cam_tf.translation();
    Eigen::Quaternionf rotation_quat =
        Eigen::Quaternionf(body_to_cam_tf.rotation());
    Eigen::Vector4f rotation = {rotation_quat.w(), rotation_quat.x(),
                                rotation_quat.y(), rotation_quat.z()};

    controller = std::make_unique<Control::VisionDWA>(
        controlType, controlLimits, maxLinearSamples, maxAngularSamples,
        robotShapeType, robotDimensions, prox_sensor_position_body,
        prox_sensor_rotation_body, translation, rotation, octreeRes,
        costWeights, maxNumThreads, config);

    // Initialize the detected boxes
    Bbox3D new_box;
    new_box.size = {0.5f, 0.5f, 1.0f};
    detected_boxes.resize(num_test_boxes);
    for (int i = 0; i < num_test_boxes; ++i) {
      auto new_box_shift =
          Eigen::Vector3f({float(0.7 * i), float(0.7 * i), 0.0f});
      auto img_frame_shift = Eigen::Vector2i({float(50 * i), float(50 * i)});
      new_box.center = new_box_shift;
      new_box.center_img_frame = img_frame_shift + ref_point_img;
      new_box.size_img_frame = {25, 25};
      new_box.timestamp = 0.0f;
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
      box.timestamp += timeStep;
    }
  };

  bool test_one_cmd_depth(const std::string image_file_path,
                          const std::vector<Bbox2D> &detections,
                          const Eigen::Vector2i &clicked_point,
                          std::vector<Path::Point> cloud) {
    // robot velocity
    Control::Velocity2D cmd;
    // Get image
    cv::Mat cv_img = cv::imread(image_file_path, cv::IMREAD_UNCHANGED);

    if (cv_img.empty()) {
      LOG_ERROR("Could not open or find the image");
    }

    // Create an Eigen matrix of type int from the OpenCV Mat
    auto depth_image = Eigen::MatrixX<unsigned short>(cv_img.rows, cv_img.cols);
    for (int i = 0; i < cv_img.rows; ++i) {
      for (int j = 0; j < cv_img.cols; ++j) {
        depth_image(i, j) = cv_img.at<unsigned short>(i, j);
      }
    }

    controller->setCameraIntrinsics(focal_length.x(), focal_length.y(),
                                    principal_point.x(), principal_point.y());

    controller->setCurrentState(robotState);
    auto found_target = controller->setInitialTracking(
        clicked_point(0), clicked_point(1), depth_image, detections);
    if (!found_target) {
      LOG_WARNING("Point not found on image");
      return false;
    } else {
      LOG_INFO("Point found on image ...");
    }

    auto res = controller->getTrackingCtrl(depth_image, detections, cmd, cloud);
    if (res.isTrajFound) {
      LOG_INFO("Got control (vx, vy, omega) = (",
               res.trajectory.velocities.vx[0], ", ",
               res.trajectory.velocities.vy[0], ", ",
               res.trajectory.velocities.omega[0], ")");
    }
    return res.isTrajFound;
  }

  bool run_test(const int numPointsPerTrajectory, std::string pltFileName,
                const bool simulate_obstacle = false) {
    Control::TrajectorySamples2D samples(2, numPointsPerTrajectory);
    Control::TrajectoryVelocities2D simulated_velocities(
        numPointsPerTrajectory);
    Control::TrajectoryPath robot_path(numPointsPerTrajectory),
        tracked_path(numPointsPerTrajectory);
    Control::Velocity2D cmd;
    Control::TrajSearchResult result;

    controller->setCurrentState(robotState);
    controller->setInitialTracking(ref_point_img(0), ref_point_img(1),
                                   detected_boxes);

    int step = 0;
    while (step < numPointsPerTrajectory) {
      Path::Point point(robotState.x, robotState.y, 0.0);
      robot_path.add(step, point);
      tracked_path.add(step, {tracked_pose.x(), tracked_pose.y(), 0.0});
      controller->setCurrentState(robotState);
      LOG_INFO("Target center at: ", tracked_pose.x(), ", ", tracked_pose.y(),
               ", ", tracked_pose.yaw());
      LOG_INFO("Robot at: ", point.x(), ", ", point.y());

      std::vector<Bbox3D> seen_boxes;
      if (use_local_frame) {
        // Transform boxes to local coordinates
        std::vector<Bbox3D> boxes_local_coordinates;
        Eigen::Isometry3f world_in_robot_tf =
            getTransformation(robotState).inverse();
        Eigen::Matrix3f abs_rotation = world_in_robot_tf.linear().cwiseAbs();
        for (auto &box : detected_boxes) {
          // Transform the box to the robot frame
          Bbox3D box_local(box);
          box_local.center = world_in_robot_tf * box.center;
          box_local.size = abs_rotation * box.size;
          boxes_local_coordinates.push_back(box_local);
        }
        seen_boxes = boxes_local_coordinates;
      } else {
        seen_boxes = detected_boxes;
      }

      // Simulate lost of target around obstacle
      if (simulate_obstacle and robotState.y > 0.2 and robotState.y < 0.4) {
        LOG_INFO("Sending EMPTY BOXES NOW!!!!!");
        seen_boxes = {};
      }
      controller->setCurrentState(robotState);
      result = controller->getTrackingCtrl(seen_boxes, cmd, cloud);

      if (result.isTrajFound) {
        LOG_INFO(FMAG("STEP: "), step,
                 FMAG(", Found best trajectory with cost: "), result.trajCost);
        if (controller->getLinearVelocityCmdX() >
            controlLimits.velXParams.maxVel) {
          LOG_ERROR(BOLD(FRED("Vx is larger than max vel: ")), KRED,
                    controller->getLinearVelocityCmdX(), RST,
                    BOLD(FRED(", Vx: ")), KRED, controlLimits.velXParams.maxVel,
                    RST);
          return false;
        }
        auto velocities = result.trajectory.velocities;
        int numSteps = 0;
        for (auto vel : velocities) {
          if (numSteps >= controlHorizon) {
            break;
          }
          numSteps++;
          cmd = vel;
          LOG_DEBUG(BOLD(FBLU("Robot at: {x: ")), KBLU, robotState.x, RST,
                    BOLD(FBLU(", y: ")), KBLU, robotState.y, RST,
                    BOLD(FBLU(", yaw: ")), KBLU, robotState.yaw, RST,
                    BOLD(FBLU("}")));

          LOG_DEBUG(BOLD(FGRN("Found Control: {Vx: ")), KGRN, vel.vx(), RST,
                    BOLD(FGRN(", Vy: ")), KGRN, vel.vy(), RST,
                    BOLD(FGRN(", Omega: ")), KGRN, vel.omega(), RST,
                    BOLD(FGRN("}")));

          robotState.update(vel, timeStep);
        }

      } else {
        LOG_ERROR(BOLD(FRED("DWA Planner failed with robot at: {x: ")), KRED,
                  robotState.x, RST, BOLD(FRED(", y: ")), KRED, robotState.y,
                  RST, BOLD(FRED(", yaw: ")), KRED, robotState.yaw, RST,
                  BOLD(FRED("}")));
        break;
      }

      tracked_pose.update(timeStep);

      moveDetectedBoxes();

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

BOOST_AUTO_TEST_CASE(Test_VisionDWA_Obstacle_Free_local) {
  // Create timer
  Timer time;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};
  bool use_local_frame = true;

  VisionDWATestConfig testConfig(cloud, use_local_frame);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_follower_with_tracker_local_frame"));
  BOOST_TEST(test_passed,
             "VisionDWA Failed To Find Control in local frame mode");
}

BOOST_AUTO_TEST_CASE(Test_VisionDWA_Obstacle_Free_global_frame) {
  // Create timer
  Timer time;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};
  bool use_local_frame = false;

  VisionDWATestConfig testConfig(cloud, use_local_frame);

  int numPointsPerTrajectory = 100;

  bool test_passed = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_follower_with_tracker_global_frame"),
      use_local_frame);
  BOOST_TEST(test_passed,
             "VisionDWA Failed To Find Control in global frame mode");
}

// BOOST_AUTO_TEST_CASE(test_VisionDWA_with_tracker_and_obstacle) {
//   // Create timer
//   Timer time;

//   // Robot pointcloud values (global frame)
//   std::vector<Path::Point> cloud = {{0.3, 0.27, 0.1}};

//   VisionDWATestConfig testConfig(cloud);

//   int numPointsPerTrajectory = 100;

//   bool test_passed = testConfig.run_test(
//       numPointsPerTrajectory,
//       std::string("vision_follower_with_tracker_and_obstacle"), true);
//   BOOST_TEST(test_passed, "VisionDWA Failed To Find Control");
// }

BOOST_AUTO_TEST_CASE(test_VisionDWA_with_depth_image) {
  // Create timer
  Timer time;

  std::string filename =
      "/home/ahr/kompass/kompass-navigation/tests/resources/control/"
      "bag_image_depth.tif";
  Bbox2D box({410, 0}, {410, 390});
  std::vector<Bbox2D> detections_2d{box};

  auto initial_point = Eigen::Vector2i{610, 200};

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.3, 10.5, 0.2}};

  VisionDWATestConfig testConfig(cloud);

  auto test_passed = testConfig.test_one_cmd_depth(filename, detections_2d,
                                                   initial_point, cloud);

  BOOST_TEST(test_passed, "VisionDWA Failed To Find Control Using Depth Image");
}
