#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "datatypes/trajectory.h"
#include "json_export.h"
#include "test.h"
#include "utils/logger.h"
#include "vision/tracker.h"
#define BOOST_TEST_MODULE KOMPASS TESTS
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <memory>
#include <random>

using namespace Kompass;

struct VisionTrackingTestConfig {
  float time_step;
  int maxSteps;
  Eigen::Vector3f target_ref_vel;
  std::unique_ptr<FeatureBasedBboxTracker> tracker;
  Control::TrajectoryPath reference_path, measured_path, tracked_path;
  Path::State reference_state;
  std::vector<Bbox3D> detected_boxes;
  std::string pltFileName = "vision_tracker_test";
  std::mt19937 gen;
  std::normal_distribution<float> noise;

  VisionTrackingTestConfig(const int &maxSteps = 50,
                           const float &time_step = 0.1,
                           const float &e_pos = 0.1, const float &e_vel = 0.1,
                           const float &e_acc = 0.1,
                           const Eigen::Vector3f target_pose = {0.0f, 0.0f, 0.0f},
                           const Eigen::Vector3f target_box_size = {0.5f, 0.5f,
                                                                    1.0f},
                           const int num_test_boxes = 3, const float target_v = 0.2, const float target_omega = 0.3)
      : time_step(time_step), maxSteps(maxSteps) {
    tracker = std::make_unique<FeatureBasedBboxTracker>(time_step, e_pos, e_vel,
                                                        e_acc);
    Bbox3D tracked_box;
    tracked_box.center = target_pose;
    tracked_box.size = target_box_size;
    tracked_box.timestamp = 0.0f;
    tracker->setInitialTracking(tracked_box);
    reference_path = Control::TrajectoryPath(maxSteps);
    tracked_path = Control::TrajectoryPath(maxSteps);
    measured_path = Control::TrajectoryPath(maxSteps);
    reference_state = Path::State(target_pose.x(), target_pose.y());
    detected_boxes.resize(num_test_boxes);
    detected_boxes[0] = tracked_box;
    Bbox3D new_box;
    new_box.size = target_box_size;
    for(int i=1; i < num_test_boxes; ++i){
      auto new_box_shift = Eigen::Vector3f({float(0.7 * i), float(0.7 * i), 0.0f});
      new_box.center = new_box_shift + target_pose;
      new_box.timestamp = 0.0f;
      detected_boxes[i] = new_box;
    }

    target_ref_vel = {target_v * cos(target_omega), target_v * sin(target_omega), 0.0};

    // Random noise generator
    std::random_device rd; // Obtain a random number from hardware
    gen = std::mt19937(rd());
    // Define the normal distribution with mean and standard deviation
    noise = std::normal_distribution<float>(0.0f, 0.01f);
  };

  void moveDetectedBoxes(const int step){
    // Update the detected boxes using the velocity command with additional measurement noise
    for(auto &box: detected_boxes){
      Eigen::Vector3f noise_vector(noise(gen), noise(gen), noise(gen));
      box.center += target_ref_vel * time_step + noise_vector;
      box.timestamp += time_step;
    }
    // Update the reference state using same velocity command (real)
    reference_state.x += target_ref_vel.x() * time_step;
    reference_state.y += target_ref_vel.y() * time_step;
    Path::Point point(reference_state.x, reference_state.y, 0.0);
    reference_path.add(
        step, Path::Point(reference_state.x, reference_state.y, 0.0));
  };
};


BOOST_AUTO_TEST_CASE(test_Vision_Tracker) {
  // Create timer
  Timer time;
  VisionTrackingTestConfig config = VisionTrackingTestConfig();
  Control::TrajectorySamples2D samples(3, config.maxSteps);
  Control::TrajectoryVelocities2D simulated_velocities(config.maxSteps);

  int step = 0;
  float tracking_error = 0.0;
  while (step < config.maxSteps) {
    // Sed detected boxes and ask tracker to update
    config.tracker->updateTracking(config.detected_boxes);
    auto measured_track = config.tracker->getRawTracking();
    auto tracked_state = config.tracker->getTrackedState();
    if(tracked_state){
      auto mat = tracked_state->col(0);
      LOG_INFO("Real target location at", config.reference_state.x, ", ", config.reference_state.y);
      LOG_INFO("Found tracked target at", mat(0), ", ", mat(1));
      tracking_error +=
          Path::Path::distance(mat.segment(0, 3), Path::Point(config.reference_state.x,
                                                   config.reference_state.y, 0.0));
      config.tracked_path.add(step, mat.segment(0, 3));
      config.measured_path.add(step, Path::Point(measured_track->box.center.x(),
                                                 measured_track->box.center.y(),
                                                 0.0));
      if (step < config.maxSteps - 1)
        simulated_velocities.add(step, Control::Velocity2D());
    }
    else{
        LOG_ERROR("Lost tracked target at step: ", step);
        break;
    }
    // Move the detected boxes one step using the reference velocity
    config.moveDetectedBoxes(step);
    step++;
  }

  // End Tracking error
  tracking_error /= config.maxSteps;
  LOG_INFO("Average Error Used Noisy Measurements = ", tracking_error);
  BOOST_TEST(tracking_error <= 0.1 , "Tracking error is larger than 10%");

  samples.push_back(simulated_velocities, config.reference_path);
  samples.push_back(simulated_velocities, config.measured_path);
  samples.push_back(simulated_velocities, config.tracked_path);

  // Plot the trajectories (Save to json then run python script for plotting)
  boost::filesystem::path executablePath = boost::dll::program_location();
  std::string file_location = executablePath.parent_path().string();

  std::string trajectories_filename = file_location + "/" + config.pltFileName;

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
}
