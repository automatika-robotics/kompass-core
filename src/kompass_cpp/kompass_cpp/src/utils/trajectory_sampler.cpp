#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/threadpool.h"
#include "utils/trajectory_sampler.h"

namespace Kompass {

namespace Control {

// Mutex for trajectory generation
static std::mutex s_trajMutex;

// TODO: Add option for OMNI robot samples generation: Rotate and move at the
// same time (ON/OFF). Current implementation does not support rotation + linear
// movement at the same time. Current implementation (for OMNI and DIFF DRIVE):
// rotate then move
TrajectorySampler::TrajectorySampler(
    ControlLimitsParams controlLimits, ControlType controlType, double timeStep,
    double predictionHorizon, double controlHorizon, int maxLinearSamples,
    int maxAngularSamples, const CollisionChecker::ShapeType robotShapeType,
    const std::vector<float> robotDimensions,
    const std::array<float, 3> &sensor_position_body,
    const std::array<float, 4> &sensor_rotation_body, const double octreeRes, const int maxNumThreads) {
  // Setup the collision checker
  collChecker = new CollisionChecker(robotShapeType, robotDimensions,
                                     sensor_position_body, sensor_rotation_body,
                                     octreeRes);

  // Setup configuration parameters
  ctrlimits = controlLimits;
  ctrType = controlType;
  time_step_ = timeStep;
  max_time_ = predictionHorizon;
  control_time_ = controlHorizon;
  lin_samples_max_ = maxLinearSamples;
  ang_samples_max_ = maxAngularSamples;

  if (ctrType != ControlType::OMNI) {
    // Discard Vy limits to eliminate movement on Y axis
    ctrlimits.velYParams = LinearVelocityControlParams(0.0, 0.0, 0.0);
  }
  this->maxNumThreads = maxNumThreads;
}

TrajectorySampler::TrajectorySampler(
    TrajectorySamplerParameters config, ControlLimitsParams controlLimits,
    ControlType controlType, const CollisionChecker::ShapeType robotShapeType,
    const std::vector<float> robotDimensions,
    const std::array<float, 3> &sensor_position_body,
    const std::array<float, 4> &sensor_rotation_body, const int maxNumThreads) {
  double octreeRes = config.getParameter<double>("octree_map_resolution");
  // Setup the collision checker
  collChecker = new CollisionChecker(robotShapeType, robotDimensions,
                                     sensor_position_body, sensor_rotation_body,
                                     octreeRes);

  // Setup configuration parameters
  ctrlimits = controlLimits;
  ctrType = controlType;

  updateParams(config);

  if (ctrType != ControlType::OMNI) {
    // Discard Vy limits to eliminate movement on Y axis
    ctrlimits.velYParams = LinearVelocityControlParams(0.0, 0.0, 0.0);
  }
  this->maxNumThreads = maxNumThreads;
}

TrajectorySampler::~TrajectorySampler() { delete collChecker; }

void TrajectorySampler::updateParams(TrajectorySamplerParameters config) {
  time_step_ = config.getParameter<double>("time_step");
  max_time_ = config.getParameter<double>("prediction_horizon");
  control_time_ = config.getParameter<double>("control_horizon");
  lin_samples_max_ = config.getParameter<int>("max_linear_samples");
  ang_samples_max_ = config.getParameter<int>("max_angular_samples");
}

void TrajectorySampler::getAdmissibleTrajsFromVel(
    const Velocity &vel, const Path::State &start_pose,
    std::vector<Trajectory> *admissible_velocity_trajectories) {

  if (std::abs(vel.vx) < MIN_VEL and std::abs(vel.vy) < MIN_VEL and
      std::abs(vel.omega) < MIN_VEL) {
    return;
  }
  Path::State simulated_pose = start_pose;
  std::vector<Velocity> simulated_velocities;
  Path::Path path;
  path.points.push_back(Path::Point(start_pose.x, start_pose.y));
  bool is_collision = false;

  for (double t = 0; t < max_time_; t += time_step_) {
    simulated_pose.x +=
        (vel.vx * cos(simulated_pose.yaw) - vel.vy * sin(simulated_pose.yaw)) *
        time_step_;
    simulated_pose.y +=
        (vel.vx * sin(simulated_pose.yaw) + vel.vy * cos(simulated_pose.yaw)) *
        time_step_;
    simulated_pose.yaw += vel.omega * time_step_;

    // Update the position of the robot in the collision checker (updates the
    // robot collision object) No need to update the Octree (laserscan)
    // collision object as the sensor data is the same
    if (maxNumThreads > 1) {
      is_collision = collChecker->checkCollisions(simulated_pose);
    } else {
      collChecker->updateState(simulated_pose);
      is_collision = collChecker->checkCollisions();
    }

    if (is_collision) {
      // LOG_DEBUG("Detected collision -> dropping sample at x: ",
      // simulated_pose.x,
      //           ", y: ", simulated_pose.y, "with Vx: ", vel.vx,
      //           ", Vy: ", vel.vy, ", Omega: ", vel.omega);
      break;
    }

    path.points.push_back(Path::Point(simulated_pose.x, simulated_pose.y));
    simulated_velocities.push_back(vel);
  }

  if (!is_collision) {
    if (maxNumThreads > 1) {
      std::lock_guard<std::mutex> lock(s_trajMutex);
      admissible_velocity_trajectories->push_back({simulated_velocities, path});
    } else {
      admissible_velocity_trajectories->push_back({simulated_velocities, path});
    }
  }
  return;
}

void TrajectorySampler::getAdmissibleTrajsFromVelDiffDrive(
    const Velocity &vel, const Path::State &start_pose,
    std::vector<Trajectory> *admissible_velocity_trajectories) {

  if (std::abs(vel.vx) < MIN_VEL and std::abs(vel.vy) < MIN_VEL and
      std::abs(vel.omega) < MIN_VEL) {
    return;
  }
  Path::State simulated_pose = start_pose;
  Path::Path path;
  path.points.push_back(Path::Point(start_pose.x, start_pose.y));
  bool is_collision = false;
  std::vector<Velocity> simulated_velocities;

  for (double t = 0; t < max_time_; t += time_step_) {

    // Alternate between linear and angular movement
    if (int(t / time_step_) % 2 == 0) {
      simulated_pose.yaw += vel.omega * time_step_;
      simulated_velocities.push_back(
          Velocity(0.0, 0.0, vel.omega, vel.steer_ang));
    } else {
      simulated_pose.x += (vel.vx * cos(simulated_pose.yaw) -
                           vel.vy * sin(simulated_pose.yaw)) *
                          time_step_;
      simulated_pose.y += (vel.vx * sin(simulated_pose.yaw) +
                           vel.vy * cos(simulated_pose.yaw)) *
                          time_step_;
      simulated_velocities.push_back(Velocity(vel.vx, vel.vy, 0.0, 0.0));
    }

    // Update the position of the robot in the collision checker (updates the
    // robot collision object) No need to update the Octree (laserscan)
    // collision object as the sensor data is the same
    if (maxNumThreads > 1) {
      is_collision = collChecker->checkCollisions(simulated_pose);
    } else {
      collChecker->updateState(simulated_pose);
      is_collision = collChecker->checkCollisions();
    }

    if (is_collision) {
      break;
    }

    path.points.push_back(Path::Point(simulated_pose.x, simulated_pose.y));
  }

  if (!is_collision) {
    if (maxNumThreads > 1) {
      std::lock_guard<std::mutex> lock(s_trajMutex);
      admissible_velocity_trajectories->push_back({simulated_velocities, path});
    } else {
      admissible_velocity_trajectories->push_back({simulated_velocities, path});
    }
  }
  return;
}

std::vector<Trajectory> TrajectorySampler::generateTrajectoriesAckermann(
    const Velocity &current_vel, const Path::State &current_pose) {
  std::vector<Trajectory> admissible_velocity_trajectories;
  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    // Implements generating smooth arc-like trajectories within the admissable
    // speed limits
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      if (std::abs(vx) >= MIN_VEL) {
        for (double omega = min_omega_; omega <= max_omega_;
             omega += ang_sample_resolution_) {

          Velocity vel = Velocity(vx, 0.0, omega); // Limit Y movement
          // Get admissible trajectories in seperate threads
          pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this, vel,
                       current_pose, &admissible_velocity_trajectories);
        }
      }
    }
  } else {
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      if (std::abs(vx) >= MIN_VEL) {
        for (double omega = min_omega_; omega <= max_omega_;
             omega += ang_sample_resolution_) {

          Velocity vel = Velocity(vx, 0.0, omega); // Limit Y movement
          getAdmissibleTrajsFromVel(vel, current_pose,
                                    &admissible_velocity_trajectories);
        }
      }
    }
  }
  return admissible_velocity_trajectories;
}

std::vector<Trajectory> TrajectorySampler::generateTrajectoriesDiffDrive(
    const Velocity &current_vel, const Path::State &current_pose) {

  std::vector<Trajectory> admissible_velocity_trajectories;
  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);

    // Generate forward/backward trajectories
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      Velocity vel = Velocity(vx, 0.0, 0.0); // Limit Y movement
      // Get admissible trajectories in seperate threads
      pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this, vel,
                   current_pose, &admissible_velocity_trajectories);
      // Generate rotation trajectories (Rotate then move)
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        Velocity vel = Velocity(vx, 0.0, omega); // Limit Y movement
        // Get admissible trajectories in seperate threads
        pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVelDiffDrive,
                     this, vel, current_pose,
                     &admissible_velocity_trajectories);
      }
    }
  } else {
    // Generate forward/backward trajectories
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      Velocity vel = Velocity(vx, 0.0, 0.0); // Limit Y movement
      getAdmissibleTrajsFromVel(vel, current_pose,
                                &admissible_velocity_trajectories);
      // Generate rotation trajectories (Rotate then move)
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        Velocity vel = Velocity(vx, 0.0, omega); // Limit Y movement
        getAdmissibleTrajsFromVelDiffDrive(vel, current_pose,
                                           &admissible_velocity_trajectories);
      }
    }
  }
  return admissible_velocity_trajectories;
}

std::vector<Trajectory>
TrajectorySampler::generateTrajectoriesOmni(const Velocity &current_vel,
                                            const Path::State &current_pose) {
  std::vector<Trajectory> admissible_velocity_trajectories;
  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    // Generate forward/backward trajectories
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      Velocity vel = Velocity(vx, 0.0, 0.0);
      // Get admissible trajectories in seperate threads
      pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this, vel,
                   current_pose, &admissible_velocity_trajectories);
    }

    // Generate lateral left/right trajectories
    for (double vy = min_vy_; vy <= max_vy_; vy += lin_sample_y_resolution_) {
      Velocity vel = Velocity(0.0, vy, 0.0);
      // Get admissible trajectories in seperate threads
      pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this, vel,
                   current_pose, &admissible_velocity_trajectories);
    }

    // Generate rotation trajectories (Rotate then move forward)
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        Velocity vel = Velocity(vx, 0.0, omega);
        // Get admissible trajectories in seperate threads
        pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVelDiffDrive,
                     this, vel, current_pose,
                     &admissible_velocity_trajectories);
      }
    }

    // Generate rotation trajectories (Rotate then move forward)
    for (double vy = min_vy_; vy <= max_vy_; vy += lin_sample_y_resolution_) {
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        Velocity vel = Velocity(0.0, vy, omega);
        // Get admissible trajectories in seperate threads
        pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVelDiffDrive,
                     this, vel, current_pose,
                     &admissible_velocity_trajectories);
      }
    }
  } else {

    // Generate forward/backward trajectories
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      Velocity vel = Velocity(vx, 0.0, 0.0);
      getAdmissibleTrajsFromVel(vel, current_pose,
                                &admissible_velocity_trajectories);
    }

    // Generate lateral left/right trajectories
    for (double vy = min_vy_; vy <= max_vy_; vy += lin_sample_y_resolution_) {
      Velocity vel = Velocity(0.0, vy, 0.0);
      getAdmissibleTrajsFromVel(vel, current_pose,
                                &admissible_velocity_trajectories);
    }

    // Generate rotation trajectories (Rotate then move forward)
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        Velocity vel = Velocity(vx, 0.0, omega);
        getAdmissibleTrajsFromVelDiffDrive(vel, current_pose,
                                           &admissible_velocity_trajectories);
      }
    }

    // Generate rotation trajectories (Rotate then move forward)
    for (double vy = min_vy_; vy <= max_vy_; vy += lin_sample_y_resolution_) {
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        Velocity vel = Velocity(0.0, vy, omega);
        getAdmissibleTrajsFromVelDiffDrive(vel, current_pose,
                                           &admissible_velocity_trajectories);
      }
    }
  }
  return admissible_velocity_trajectories;
}

std::vector<Trajectory>
TrajectorySampler::getNewTrajectories(const Velocity &current_vel,
                                      const Path::State &current_pose) {
  // Get the range of reachable velocities from the current velocity
  UpdateReachableVelocityRange(current_vel);

  switch (ctrType) {
  case ControlType::ACKERMANN:
    return generateTrajectoriesAckermann(current_vel, current_pose);
  case ControlType::DIFFERENTIAL_DRIVE:
    return generateTrajectoriesDiffDrive(current_vel, current_pose);
  case ControlType::OMNI:
    return generateTrajectoriesOmni(current_vel, current_pose);
  default:
    return std::vector<Trajectory>{};
  }
}

std::vector<Trajectory>
TrajectorySampler::generateTrajectories(const Velocity &current_vel,
                                        const Path::State &current_pose,
                                        const Control::LaserScan &scan) {
  collChecker->updateState(current_pose);
  // Update the laserscan values at the current location -> no need to update
  // it as scan is not changing (during the simulation)
  collChecker->updateScan(scan.ranges, scan.angles);
  return getNewTrajectories(current_vel, current_pose);
}

std::vector<Trajectory>
TrajectorySampler::generateTrajectories(const Velocity &current_vel,
                                        const Path::State &current_pose,
                                        const std::vector<Point3D> &cloud) {
  collChecker->updateState(current_pose);
  // Update the PointCloud values
  collChecker->updatePointCloud(cloud);
  return getNewTrajectories(current_vel, current_pose);
}

void TrajectorySampler::UpdateReachableVelocityRange(
    Control::Velocity currentVel) {

  max_vx_ = std::min(ctrlimits.velXParams.maxVel,
                     currentVel.vx +
                         ctrlimits.velXParams.maxAcceleration * max_time_);
  min_vx_ = std::max(-ctrlimits.velXParams.maxVel,
                     currentVel.vx -
                         ctrlimits.velXParams.maxDeceleration * max_time_);

  if (ctrType == ControlType::OMNI) {
    max_vy_ = std::min(ctrlimits.velYParams.maxVel,
                       currentVel.vy +
                           ctrlimits.velYParams.maxAcceleration * max_time_);
    min_vy_ = std::max(-ctrlimits.velYParams.maxVel,
                       currentVel.vy -
                           ctrlimits.velYParams.maxDeceleration * max_time_);
  } else {
    max_vy_ = 0.0;
    min_vy_ = 0.0;
  }

  // Step cannot be zero -> creates infinite loop
  lin_sample_x_resolution_ =
      std::max((max_vx_ - min_vx_) / lin_samples_max_, 0.001);
  lin_sample_y_resolution_ =
      std::max((max_vy_ - min_vy_) / lin_samples_max_, 0.001);

  max_omega_ = std::min(ctrlimits.omegaParams.maxOmega,
                        currentVel.omega +
                            ctrlimits.omegaParams.maxAcceleration * max_time_);
  min_omega_ = std::max(-ctrlimits.omegaParams.maxOmega,
                        currentVel.omega -
                            ctrlimits.omegaParams.maxDeceleration * max_time_);

  ang_sample_resolution_ =
      std::max((max_omega_ - min_omega_) / ang_samples_max_, 0.001);
}

}; // namespace Control
} // namespace Kompass
