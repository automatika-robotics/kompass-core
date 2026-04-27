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

  double distance_tolerance = 0.1;
  double target_distance = 0.2;
  float robot_radius = 0.1;

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
                      const int predictionHorizon = 20,
                      const int controlHorizon = 2,
                      const int maxLinearSamples = 20,
                      const int maxAngularSamples = 20,
                      const float maxVel = 2.0, const float maxOmega = 4.0,
                      const int maxNumThreads = 1,
                      const double reference_path_distance_weight = 1.0,
                      const double goal_distance_weight = 0.5,
                      const double obstacles_distance_weight = 0.0)
      : use_local_frame(use_local_frame), timeStep(timeStep),
        predictionHorizon(predictionHorizon), controlHorizon(controlHorizon),
        maxLinearSamples(maxLinearSamples),
        maxAngularSamples(maxAngularSamples), maxNumThreads(maxNumThreads),
        octreeRes(0.1), x_params(maxVel, 5.0, 10.0), y_params(1, 3, 5),
        angular_params(3.14, maxOmega, 3.0, 3.0),
        controlLimits(x_params, y_params, angular_params),
        controlType(Control::ControlType::DIFFERENTIAL_DRIVE),
        robotShapeType(Kompass::CollisionChecker::ShapeType::CYLINDER),
        robotDimensions{robot_radius, 0.4},
        prox_sensor_position_body{0.0, 0.0, 0.0},
        prox_sensor_rotation_body{0, 0, 0, 1}, cloud(sensor_points),
        robotState(-0.8, 0.0, 0.0, 0.0), tracked_vel(0.1, 0.0, 0.1),
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
    config.setParameter("target_distance", target_distance);
    config.setParameter("target_orientation", 0.0);
    config.setParameter("distance_tolerance", distance_tolerance);

    // Body to (FLU-aligned) camera tf from robot of test pictures.
    // The DepthDetector applies the optical->FLU rotation internally, so this
    // tf must be the body->camera-link transform, NOT the body->optical frame.
    auto body_to_link_tf =
        getTransformation(Eigen::Quaternionf{0.0f, 0.1987f, 0.0f, 0.98f},
                          Eigen::Vector3f{0.32f, 0.0209f, 0.3f});

    auto link_to_cam_tf =
        getTransformation(Eigen::Quaternionf{0.01f, -0.00131f, 0.002f, 0.9999f},
                          Eigen::Vector3f{0.0f, 0.0105f, 0.0f});

    Eigen::Isometry3f body_to_cam_tf = body_to_link_tf * link_to_cam_tf;

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
      auto img_frame_shift = Eigen::Vector2i{50 * i, 50 * i};
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

  // Build a synthetic 2D laserscan in the robot body frame whose returns
  // come from the tracked target's circular footprint. The target center
  // is taken from `tracked_pose` (world frame) and projected into the
  // robot body frame using `robotState`. Rays that don't hit the target
  // return `max_range`. Used to feed the LaserScan path of getTrackingCtrl.
  Control::LaserScan generateScanFromTarget(float max_range = 5.0f,
                                            int num_rays = 360) const {
    // Target footprint radius from the tracked bbox extents
    float target_radius = 0.0f;
    if (!detected_boxes.empty()) {
      target_radius = 0.5f * std::max(detected_boxes.front().size.x(),
                                      detected_boxes.front().size.y());
    }

    // World-to-body for the target center
    const float yaw = float(robotState.yaw);
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);
    const float dx = float(tracked_pose.x() - robotState.x);
    const float dy = float(tracked_pose.y() - robotState.y);
    const float tx = c * dx + s * dy;
    const float ty = -s * dx + c * dy;

    std::vector<double> ranges, angles;
    ranges.reserve(num_rays);
    angles.reserve(num_rays);
    const double r_sq = double(target_radius) * double(target_radius);
    for (int i = 0; i < num_rays; ++i) {
      const double angle = -M_PI + (2.0 * M_PI * i) / num_rays;
      const double ca = std::cos(angle);
      const double sa = std::sin(angle);
      const double along = tx * ca + ty * sa;
      const double perp = std::abs(tx * sa - ty * ca);
      double range = max_range;
      if (along > 0.0 && perp <= target_radius) {
        const double half_chord = std::sqrt(std::max(0.0, r_sq - perp * perp));
        const double r_hit = along - half_chord;
        if (r_hit > 0.0 && r_hit < range) {
          range = r_hit;
        }
      }
      ranges.push_back(range);
      angles.push_back(angle);
    }
    return Control::LaserScan(std::move(ranges), std::move(angles));
  };

  // Build a ring of points sitting on the tracked target's circular
  // footprint (radius = 0.5 * max(box.size.x, box.size.y), matching the
  // inscribed-circle convention used elsewhere). Returned in world frame —
  // the caller is responsible for any frame conversion.
  std::vector<Path::Point> targetBodyPoints(int num_points = 24) const {
    const double r = 0.5 * std::max(detected_boxes.front().size.x(),
                                    detected_boxes.front().size.y());
    std::vector<Path::Point> pts;
    pts.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
      const double a = (2.0 * M_PI * i) / num_points;
      pts.emplace_back(tracked_pose.x() + r * std::cos(a),
                       tracked_pose.y() + r * std::sin(a), 0.1);
    }
    return pts;
  }

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

  double run_test(const int numPointsPerTrajectory, std::string pltFileName,
                  const bool simulate_obstacle = false,
                  const bool include_target_in_cloud = false) {
    Control::TrajectorySamples2D samples(2, numPointsPerTrajectory);
    Control::TrajectoryVelocities2D simulated_velocities(
        numPointsPerTrajectory);
    Control::TrajectoryPath robot_path(numPointsPerTrajectory),
        tracked_path(numPointsPerTrajectory);
    // Zero-fill the pre-allocated Eigen vectors so any slots that remain
    // unfilled (early break from the divergence guard, etc.) plot as zeros
    // instead of uninitialized garbage / NaN.
    robot_path.x.setZero();
    robot_path.y.setZero();
    robot_path.z.setZero();
    tracked_path.x.setZero();
    tracked_path.y.setZero();
    tracked_path.z.setZero();
    simulated_velocities.vx.setZero();
    simulated_velocities.vy.setZero();
    simulated_velocities.omega.setZero();
    Control::Velocity2D cmd;
    Control::TrajSearchResult result;

    controller->setCurrentState(robotState);

    // Set the initial tracking boxes in local or global frame
    std::vector<Bbox3D> initial_boxes;
    if (use_local_frame) {
      // Transform boxes to local coordinates for initialization
      Eigen::Isometry3f world_in_robot_tf =
          getTransformation(robotState).inverse();
      Eigen::Matrix3f abs_rotation = world_in_robot_tf.linear().cwiseAbs();

      for (auto &box : detected_boxes) {
        Bbox3D box_local(box);
        box_local.center = world_in_robot_tf * box.center;
        box_local.size = abs_rotation * box.size;
        initial_boxes.push_back(box_local);
      }
    } else {
      initial_boxes = detected_boxes;
    }

    controller->setInitialTracking(ref_point_img(0), ref_point_img(1),
                                   initial_boxes);

    int step = 0;
    double start_distance = std::sqrt(std::pow(robotState.x - tracked_pose.x(), 2) +
              std::pow(robotState.y - tracked_pose.y(), 2));
    double end_distance = start_distance;

    while (step < numPointsPerTrajectory) {
      Path::Point point(robotState.x, robotState.y, 0.0);
      robot_path.add(step, point);
      tracked_path.add(step, {tracked_pose.x(), tracked_pose.y(), 0.0});
      controller->setCurrentState(robotState);

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

      end_distance = std::sqrt(std::pow(robotState.x - tracked_pose.x(), 2) +
                               std::pow(robotState.y - tracked_pose.y(), 2));

      // NOTE: the following code simulates blocking the tracked target during navigation.
      // TODO: Add a test case with an obstacle and use the commented code to test loosing the target.
      // Simulate occlusion: blank detections when an obstacle point lies
      // within a cone between the robot and the tracked target AND the
      // obstacle is close enough to the robot to actually block the view.
      // From far away a small obstacle subtends a negligible angle.
      // if (simulate_obstacle) {
      //   Eigen::Vector2f robot_pos(robotState.x, robotState.y);
      //   Eigen::Vector2f target_pos(tracked_pose.x(), tracked_pose.y());
      //   Eigen::Vector2f to_target = target_pos - robot_pos;
      //   float dist_to_target = to_target.norm();
      //   if (dist_to_target > 1e-3f) {
      //     Eigen::Vector2f dir = to_target / dist_to_target;
      //     constexpr float occlusion_radius = 0.15f; // half-width of the cone
      //     constexpr float occlusion_range = 0.3f;   // max robot-to-obs range
      //     for (const auto &obs : cloud) {
      //       Eigen::Vector2f to_obs(obs.x() - robot_pos.x(),
      //                              obs.y() - robot_pos.y());
      //       float dist_to_obs = to_obs.norm();
      //       float proj = to_obs.dot(dir);
      //       // Obstacle must be between robot and target AND within range
      //       if (proj > 0.0f && proj < dist_to_target &&
      //           dist_to_obs < occlusion_range) {
      //         float perp = std::abs(to_obs.x() * dir.y() - to_obs.y() * dir.x());
      //         if (perp < occlusion_radius) {
      //           LOG_INFO("Obstacle occluding target — sending empty boxes");
      //           seen_boxes = {};
      //           break;
      //         }
      //       }
          // }
        // }
      // }
      controller->setCurrentState(robotState);
      // Optionally append target body returns to the cloud each step so the
      // controller sees the target as part of the sensor data (mirrors the
      // synthetic LaserScan flow in run_test_laserscan).
      std::vector<Path::Point> step_cloud = cloud;
      if (include_target_in_cloud) {
        const auto tgt_pts = targetBodyPoints();
        step_cloud.insert(step_cloud.end(), tgt_pts.begin(), tgt_pts.end());
      }
      result = controller->getTrackingCtrl(seen_boxes, cmd, step_cloud);

      if (result.isTrajFound) {
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
        LOG_ERROR(BOLD(FRED("VisionDWA Planner failed with robot at: {x: ")), KRED,
                  robotState.x, RST, BOLD(FRED(", y: ")), KRED, robotState.y,
                  RST, BOLD(FRED(", yaw: ")), KRED, robotState.yaw, RST,
                  BOLD(FRED("}")));
        break;
      }

      // Advance target and detected boxes the same number of steps as the
      // robot to keep the simulation in sync.
      int applied_steps = result.isTrajFound
                              ? std::min(controlHorizon,
                                         static_cast<int>(
                                             result.trajectory.velocities.vx.size()))
                              : 1;
      for (int s = 0; s < applied_steps; ++s) {
        tracked_pose.update(timeStep);
        moveDetectedBoxes();
      }



      if(end_distance > 3.0 * start_distance){
        LOG_ERROR("Robot is moving too far from tracked target");
        break;
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

    // Save obstacle cloud so the plot can display it
    std::string obstacles_filename = file_location + "/" + pltFileName + "_obs";
    {
      json j_obs;
      j_obs["points"] = json::array();
      for (const auto &pt : cloud) {
        j_obs["points"].push_back(json{{"x", pt.x()}, {"y", pt.y()}});
      }
      std::ofstream obs_file(obstacles_filename + ".json");
      if (obs_file.is_open()) {
        obs_file << j_obs.dump(4);
      }
    }

    std::string command = "python3 " + file_location +
                          "/trajectory_sampler_plt.py --samples \"" +
                          trajectories_filename + "\"";
    if (simulate_obstacle) {
      command += " --obstacles \"" + obstacles_filename + "\"";
    }

    // Execute the Python script
    int res = system(command.c_str());
    if (res != 0)
      throw std::system_error(res, std::generic_category(),
                              "Python script failed with error code");

    // Compute final tracking error
    return std::sqrt(std::pow(robotState.x - tracked_pose.x(), 2) +
                     std::pow(robotState.y - tracked_pose.y(), 2));
  }

  // Same simulation flow as run_test but feeds a synthetic LaserScan that
  // contains returns from the tracked target itself. Exercises the LaserScan
  // path of filterTargetFromSensorPoints — without filtering, the target's
  // own returns would block the robot from approaching to target_distance.
  double run_test_laserscan(const int numPointsPerTrajectory,
                            std::string pltFileName) {
    Control::TrajectorySamples2D samples(2, numPointsPerTrajectory);
    Control::TrajectoryVelocities2D simulated_velocities(
        numPointsPerTrajectory);
    Control::TrajectoryPath robot_path(numPointsPerTrajectory),
        tracked_path(numPointsPerTrajectory);
    robot_path.x.setZero();
    robot_path.y.setZero();
    robot_path.z.setZero();
    tracked_path.x.setZero();
    tracked_path.y.setZero();
    tracked_path.z.setZero();
    simulated_velocities.vx.setZero();
    simulated_velocities.vy.setZero();
    simulated_velocities.omega.setZero();
    Control::Velocity2D cmd;
    Control::TrajSearchResult result;

    controller->setCurrentState(robotState);

    std::vector<Bbox3D> initial_boxes;
    if (use_local_frame) {
      Eigen::Isometry3f world_in_robot_tf =
          getTransformation(robotState).inverse();
      Eigen::Matrix3f abs_rotation = world_in_robot_tf.linear().cwiseAbs();
      for (auto &box : detected_boxes) {
        Bbox3D box_local(box);
        box_local.center = world_in_robot_tf * box.center;
        box_local.size = abs_rotation * box.size;
        initial_boxes.push_back(box_local);
      }
    } else {
      initial_boxes = detected_boxes;
    }
    controller->setInitialTracking(ref_point_img(0), ref_point_img(1),
                                   initial_boxes);

    int step = 0;
    double start_distance =
        std::sqrt(std::pow(robotState.x - tracked_pose.x(), 2) +
                  std::pow(robotState.y - tracked_pose.y(), 2));
    double end_distance = start_distance;

    // Collect scan endpoints in world frame for plotting/debug.
    std::vector<Path::Point> scan_points_world;

    while (step < numPointsPerTrajectory) {
      robot_path.add(step, Path::Point(robotState.x, robotState.y, 0.0));
      tracked_path.add(step, {tracked_pose.x(), tracked_pose.y(), 0.0});
      controller->setCurrentState(robotState);

      std::vector<Bbox3D> seen_boxes;
      if (use_local_frame) {
        std::vector<Bbox3D> boxes_local_coordinates;
        Eigen::Isometry3f world_in_robot_tf =
            getTransformation(robotState).inverse();
        Eigen::Matrix3f abs_rotation = world_in_robot_tf.linear().cwiseAbs();
        for (auto &box : detected_boxes) {
          Bbox3D box_local(box);
          box_local.center = world_in_robot_tf * box.center;
          box_local.size = abs_rotation * box.size;
          boxes_local_coordinates.push_back(box_local);
        }
        seen_boxes = boxes_local_coordinates;
      } else {
        seen_boxes = detected_boxes;
      }

      end_distance = std::sqrt(std::pow(robotState.x - tracked_pose.x(), 2) +
                               std::pow(robotState.y - tracked_pose.y(), 2));

      // Generate the laserscan AFTER the robot/target positions are settled
      // for this step so the scan reflects the current geometry.
      Control::LaserScan scan = generateScanFromTarget();

      // Record scan returns that came from the target (not the max-range
      // sentinel) in world coords for visualization.
      const float max_range = 5.0f;
      const float yaw = float(robotState.yaw);
      const float cy_ = std::cos(yaw);
      const float sy_ = std::sin(yaw);
      for (size_t i = 0; i < scan.ranges.size(); ++i) {
        if (scan.ranges[i] >= max_range)
          continue;
        const double bx = scan.ranges[i] * std::cos(scan.angles[i]);
        const double by = scan.ranges[i] * std::sin(scan.angles[i]);
        const double wx = robotState.x + cy_ * bx - sy_ * by;
        const double wy = robotState.y + sy_ * bx + cy_ * by;
        scan_points_world.emplace_back(wx, wy, 0.0);
      }

      controller->setCurrentState(robotState);
      result = controller->getTrackingCtrl(seen_boxes, cmd, scan);

      if (result.isTrajFound) {
        auto velocities = result.trajectory.velocities;
        int numSteps = 0;
        for (auto vel : velocities) {
          if (numSteps >= controlHorizon) {
            break;
          }
          numSteps++;
          cmd = vel;
          robotState.update(vel, timeStep);
        }
      } else {
        LOG_ERROR(BOLD(FRED("DWA Planner failed with robot at: {x: ")), KRED,
                  robotState.x, RST, BOLD(FRED(", y: ")), KRED, robotState.y,
                  RST, BOLD(FRED(", yaw: ")), KRED, robotState.yaw, RST,
                  BOLD(FRED("}")));
        break;
      }

      int applied_steps =
          result.isTrajFound
              ? std::min(controlHorizon,
                         static_cast<int>(result.trajectory.velocities.vx.size()))
              : 1;
      for (int s = 0; s < applied_steps; ++s) {
        tracked_pose.update(timeStep);
        moveDetectedBoxes();
      }

      if (end_distance > 3.0 * start_distance) {
        LOG_ERROR("Robot is moving too far from tracked target");
        break;
      }

      step++;
    }
    samples.push_back(simulated_velocities, robot_path);
    samples.push_back(simulated_velocities, tracked_path);

    boost::filesystem::path executablePath = boost::dll::program_location();
    std::string file_location = executablePath.parent_path().string();
    std::string trajectories_filename = file_location + "/" + pltFileName;
    saveTrajectoriesToJson(samples, trajectories_filename + ".json");

    // Save scan points so the plot shows where the simulated laser hit.
    std::string obstacles_filename = file_location + "/" + pltFileName + "_obs";
    {
      json j_obs;
      j_obs["points"] = json::array();
      for (const auto &pt : scan_points_world) {
        j_obs["points"].push_back(json{{"x", pt.x()}, {"y", pt.y()}});
      }
      std::ofstream obs_file(obstacles_filename + ".json");
      if (obs_file.is_open()) {
        obs_file << j_obs.dump(4);
      }
    }

    std::string command = "python3 " + file_location +
                          "/trajectory_sampler_plt.py --samples \"" +
                          trajectories_filename + "\" --obstacles \"" +
                          obstacles_filename + "\"";
    int res = system(command.c_str());
    if (res != 0)
      throw std::system_error(res, std::generic_category(),
                              "Python script failed with error code");

    return std::sqrt(std::pow(robotState.x - tracked_pose.x(), 2) +
                     std::pow(robotState.y - tracked_pose.y(), 2));
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

  double end_distance = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_follower_with_tracker_local_frame"));

  // target_distance is the edge-to-edge gap between robot and target body,
  // so the robot's center settles at:
  //   robot_radius + target_distance + target_radius
  // target_radius matches VisionDWA's inscribed-circle convention.
  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());
  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;

  bool test_passed = std::abs(distance_error) < 2.0 * testConfig.distance_tolerance;

  if (test_passed) {
    LOG_INFO("Tracking finished successfully. End error: ", distance_error,
             " < tolerance ", 2.0 * testConfig.distance_tolerance);
  } else {
    LOG_ERROR("Tracking Failed! End error: ", distance_error, " > tolerance ",
              2.0 * testConfig.distance_tolerance);
  }

  BOOST_TEST(test_passed,
             "VisionDWA Failed To Track Target in local frame mode");
}

BOOST_AUTO_TEST_CASE(Test_VisionDWA_Obstacle_Free_global_frame) {
  // Create timer
  Timer time;

  // Robot pointcloud values (global frame)
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};
  bool use_local_frame = false;

  VisionDWATestConfig testConfig(cloud, use_local_frame);

  int numPointsPerTrajectory = 100;

  double end_distance = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_follower_with_tracker_global_frame"));

  // target_distance is the edge-to-edge gap between robot and target body,
  // so the robot's center settles at:
  //   robot_radius + target_distance + target_radius
  // target_radius matches VisionDWA's inscribed-circle convention.
  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());
  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;

  bool test_passed =
      std::abs(distance_error) < 2.0 * testConfig.distance_tolerance;

  if (test_passed) {
    LOG_INFO("Tracking finished successfully. End error: ", distance_error,
             " < tolerance ", testConfig.distance_tolerance);
  } else {
    LOG_ERROR("Tracking Failed! End error: ", distance_error, " > tolerance ",
              testConfig.distance_tolerance);
  }

  BOOST_TEST(test_passed,
             "VisionDWA Failed To Find Control in global frame mode");
}

// Small obstacle the robot can navigate around and resume tracking.
BOOST_AUTO_TEST_CASE(test_VisionDWA_small_obstacle) {
  Timer time;

  // Single obstacle point slightly above the x-axis. The robot tracks
  // the target along y≈0. As it approaches, the obstacle briefly enters
  // the robot-to-target cone causing a short occlusion. The robot's forward
  // momentum carries it past the obstacle, the cone clears, and tracking
  // resumes. This tests brief occlusion recovery, not global re-routing.
  std::vector<Path::Point> cloud = {{-0.1, 0.08, 0.1}};

  VisionDWATestConfig testConfig(
      cloud, /*use_local_frame=*/false, /*timeStep=*/0.1,
      /*predictionHorizon=*/10, /*controlHorizon=*/3,
      /*maxLinearSamples=*/20, /*maxAngularSamples=*/20,
      /*maxVel=*/2.0, /*maxOmega=*/4.0, /*maxNumThreads=*/10,
      /*reference_path_distance_weight=*/0.5,
      /*goal_distance_weight=*/1.0,
      /*obstacles_distance_weight=*/0.0);

  // Target moves straight along +x
  testConfig.tracked_vel = Control::Velocity2D(0.1, 0.0, 0.0);
  testConfig.tracked_pose =
      Control::TrackedPose2D(0.0, 0.0, 0.0, testConfig.tracked_vel);

  int numPointsPerTrajectory = 100;

  double end_distance = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_dwa_small_obstacle"), true);

  // The robot should navigate around the small obstacle and resume tracking.
  // Assert it ends up within a reasonable distance of the target.
  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());
  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;

  // allow more margin of error since there are obstacles to navigate around
  bool test_passed =
      std::abs(distance_error) < 4.0 * testConfig.target_distance;

  if (test_passed) {
    LOG_INFO("Small obstacle: tracking resumed. End error: ", distance_error,
             " < tolerance ", 4.0 * testConfig.target_distance);
  } else {
    LOG_ERROR("Small obstacle: tracking failed. End error: ", distance_error,
              " > tolerance ", 4.0 * testConfig.target_distance);
  }

  BOOST_TEST(test_passed,
             "VisionDWA failed to navigate around small obstacle");
}

// Same scenario as test_VisionDWA_small_obstacle, but the tracked target
// also shows up in the point cloud each step (a ring of points around the
// target center at the inscribed-circle radius). This exercises the
// controller's ability to follow the target without false-flagging the
// target's own returns as obstacles to avoid.
BOOST_AUTO_TEST_CASE(test_VisionDWA_small_obstacle_target_in_cloud) {
  Timer time;

  std::vector<Path::Point> cloud = {{-0.1, 0.08, 0.1}};

  VisionDWATestConfig testConfig(
      cloud, /*use_local_frame=*/false, /*timeStep=*/0.1,
      /*predictionHorizon=*/10, /*controlHorizon=*/3,
      /*maxLinearSamples=*/20, /*maxAngularSamples=*/20,
      /*maxVel=*/2.0, /*maxOmega=*/4.0, /*maxNumThreads=*/10,
      /*reference_path_distance_weight=*/0.5,
      /*goal_distance_weight=*/1.0,
      /*obstacles_distance_weight=*/0.0);

  testConfig.tracked_vel = Control::Velocity2D(0.1, 0.0, 0.0);
  testConfig.tracked_pose =
      Control::TrackedPose2D(0.0, 0.0, 0.0, testConfig.tracked_vel);

  int numPointsPerTrajectory = 100;

  double end_distance = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_dwa_small_obstacle_target_in_cloud"),
      /*simulate_obstacle=*/true,
      /*include_target_in_cloud=*/true);

  // Same loose tolerance as the obstacle-only test: the small obstacle
  // forces a detour, so we don't expect a tight settling distance.
  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());
  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;
  bool test_passed =
      std::abs(distance_error) < 4.0 * testConfig.target_distance;

  if (test_passed) {
    LOG_INFO("Small obstacle + target in cloud: tracking resumed. End error: ",
             distance_error, " < tolerance ", 4.0 * testConfig.target_distance);
  } else {
    LOG_ERROR("Small obstacle + target in cloud: tracking failed. End error: ",
              distance_error, " > tolerance ", 4.0 * testConfig.target_distance);
  }

  BOOST_TEST(test_passed, "VisionDWA failed to navigate around small obstacle "
                         "with target body in pointcloud");
}

// Two obstacles, one on each side of the target's path. The robot should
// follow a doubled-curve trajectory: detour first one way, then the other,
// while resuming tracking between the two avoidance maneuvers.
BOOST_AUTO_TEST_CASE(test_VisionDWA_two_sided_obstacles) {
  Timer time;

  // Target moves straight along +x at y=0. First obstacle above the path,
  // second obstacle below the path, separated along x so the robot has time
  // to recover tracking between them.
  std::vector<Path::Point> cloud = {
      {0.3, 0.08, 0.1},   // first obstacle above the path
      {0.9, -0.08, 0.1},  // second obstacle below the path
  };

  VisionDWATestConfig testConfig(
      cloud, /*use_local_frame=*/false, /*timeStep=*/0.1,
      /*predictionHorizon=*/10, /*controlHorizon=*/3,
      /*maxLinearSamples=*/20, /*maxAngularSamples=*/20,
      /*maxVel=*/2.0, /*maxOmega=*/4.0, /*maxNumThreads=*/10,
      /*reference_path_distance_weight=*/1.0,
      /*goal_distance_weight=*/0.5,
      /*obstacles_distance_weight=*/0.0);

  testConfig.tracked_vel = Control::Velocity2D(0.1, 0.0, 0.0);
  testConfig.tracked_pose =
      Control::TrackedPose2D(0.0, 0.0, 0.0, testConfig.tracked_vel);

  int numPointsPerTrajectory = 100;

  double end_distance = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_dwa_two_sided_obstacles"), true);

  // The robot navigates around both obstacles and resumes tracking. Allow a
  // looser tolerance than the obstacle-free case since each detour adds error.
  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());
  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;

  // allow more margin of error since there are two obstacles to navigate around
  bool test_passed =
      std::abs(distance_error) < 3.0 *testConfig.target_distance;

  if (test_passed) {
    LOG_INFO("Two-sided obstacles: tracking resumed. End error: ",
             distance_error, " < tolerance ", 3.0 * testConfig.target_distance);
  } else {
    LOG_ERROR("Two-sided obstacles: tracking failed. End error: ",
              distance_error, " > tolerance ", 3.0 * testConfig.target_distance);
  }

  BOOST_TEST(test_passed,
             "VisionDWA failed to navigate around two-sided obstacles");
}

// Wide wall directly between the robot and the target. The robot cannot
// route around the wall with a local planner, so it must STOP rather than
// drive into the wall or oscillate dangerously. Success criterion:
//   1. The robot does not get any closer to the wall than its radius.
//   2. The robot does not drift away uncontrollably (graceful stop).
BOOST_AUTO_TEST_CASE(test_VisionDWA_blocking_wall) {
  Timer time;

  // Wall at x=0.3 spanning y ∈ [-0.5, 0.5]: too wide to circumnavigate.
  std::vector<Path::Point> cloud;
  for (float y = -0.5f; y <= 0.5f; y += 0.05f) {
    cloud.push_back({0.3, y, 0.1});
  }

  VisionDWATestConfig testConfig(
      cloud, /*use_local_frame=*/false, /*timeStep=*/0.1,
      /*predictionHorizon=*/10, /*controlHorizon=*/3,
      /*maxLinearSamples=*/20, /*maxAngularSamples=*/20,
      /*maxVel=*/2.0, /*maxOmega=*/4.0, /*maxNumThreads=*/10,
      /*reference_path_distance_weight=*/1.0,
      /*goal_distance_weight=*/0.5,
      /*obstacles_distance_weight=*/0.0);

  testConfig.tracked_vel = Control::Velocity2D(0.1, 0.0, 0.0);
  testConfig.tracked_pose =
      Control::TrackedPose2D(0.0, 0.0, 0.0, testConfig.tracked_vel);

  int numPointsPerTrajectory = 100;

  double end_distance = testConfig.run_test(
      numPointsPerTrajectory,
      std::string("vision_dwa_blocking_wall"), true);

  // Robot must stop before the wall (not collide with it).
  // The wall is at x=0.3; with robot_radius=0.1 the robot center must stay
  // at x < (0.3 - robot_radius) = 0.2.
  const double wall_x = 0.3;
  const double min_safe_distance = wall_x - testConfig.robot_radius;
  bool stopped_before_wall = testConfig.robotState.x < min_safe_distance;

  // Also verify the robot did not drift far off the path.
  bool stayed_in_vicinity = std::abs(testConfig.robotState.y) < 0.5;

  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());

  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;

  bool test_passed = (stopped_before_wall && stayed_in_vicinity) ||
                     (std::abs(distance_error) < 2.0 * testConfig.distance_tolerance);

  if (test_passed) {
    LOG_INFO("Blocking wall: robot stopped safely at (",
             testConfig.robotState.x, ", ", testConfig.robotState.y,
             "). End distance to target: ", end_distance);
  } else {
    LOG_ERROR("Blocking wall: robot ended at (", testConfig.robotState.x,
              ", ", testConfig.robotState.y,
              ") — stopped_before_wall=", stopped_before_wall,
              ", stayed_in_vicinity=", stayed_in_vicinity);
  }

  BOOST_TEST(test_passed,
             "VisionDWA failed to stop safely before a blocking wall");
}

// The tracked target's own laserscan returns appear in the sensor data fed
// to the DWA planner. Without filterTargetFromSensorPoints, those returns
// would block the planner from approaching to target_distance — every
// trajectory that gets within a robot radius of the target would be marked
// as a collision. With the filter, the target's returns are removed and the
// robot can settle at target_distance.
BOOST_AUTO_TEST_CASE(test_VisionDWA_target_in_laserscan) {
  Timer time;

  // Cloud is unused in this test; the synthetic LaserScan is built each step
  // from the current robot/target geometry inside run_test_laserscan.
  std::vector<Path::Point> cloud = {{10.0, 10.0, 0.1}};
  bool use_local_frame = true;

  VisionDWATestConfig testConfig(
      cloud, use_local_frame, /*timeStep=*/0.1,
      /*predictionHorizon=*/10, /*controlHorizon=*/3,
      /*maxLinearSamples=*/20, /*maxAngularSamples=*/20,
      /*maxVel=*/2.0, /*maxOmega=*/4.0, /*maxNumThreads=*/10,
      /*reference_path_distance_weight=*/1.0,
      /*goal_distance_weight=*/0.5,
      /*obstacles_distance_weight=*/0.0);

  // Static target so the test focuses purely on filter correctness, not
  // tracking dynamics.
  testConfig.tracked_vel = Control::Velocity2D(0.0, 0.0, 0.0);
  testConfig.tracked_pose =
      Control::TrackedPose2D(0.0, 0.0, 0.0, testConfig.tracked_vel);

  int numPointsPerTrajectory = 100;

  double end_distance = testConfig.run_test_laserscan(
      numPointsPerTrajectory,
      std::string("vision_dwa_target_in_laserscan"));

  // target_distance is now the edge-to-edge gap between robot and target,
  // so the robot's center stops at:
  //   robot_radius + target_distance + target_radius
  // from the target's center. target_radius mirrors
  // VisionDWA::currentTargetRadius: 0.5 * max(box.size.x, box.size.y).
  const double target_radius =
      0.5 * std::max(testConfig.detected_boxes.front().size.x(),
                     testConfig.detected_boxes.front().size.y());

  double distance_error = end_distance - testConfig.robot_radius -
                          testConfig.target_distance - target_radius;

  bool test_passed = std::abs(distance_error) <= 2.0 * testConfig.distance_tolerance;

  if (test_passed) {
    LOG_INFO("Target-in-laserscan: tracking succeeded. End error: ",
             distance_error, " < tolerance ", testConfig.distance_tolerance,
             " (target_radius=", target_radius, ")");
  } else {
    LOG_ERROR("Target-in-laserscan: tracking failed. End error: ",
              distance_error, " > tolerance ", testConfig.distance_tolerance,
              " (target_radius=", target_radius, ")");
  }

  BOOST_TEST(test_passed,
             "VisionDWA failed to settle at target_distance gap from the "
             "tracked target's edge");
}
