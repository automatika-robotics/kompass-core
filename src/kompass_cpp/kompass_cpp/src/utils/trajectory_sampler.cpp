#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/logger.h"
#include "utils/threadpool.h"
#include "utils/trajectory_sampler.h"
namespace Kompass {

namespace Control {

// Mutex for trajectory generation
static std::mutex s_trajMutex;

// TODO: Add option for OMNI robot samples generation: Rotate and move at the
// same time (ON/OFF). Current implementation does not support rotation + linear
// movement at the same time.
TrajectorySampler::TrajectorySampler(
    ControlLimitsParams controlLimits, ControlType controlType, double timeStep,
    double predictionHorizon, double controlHorizon, int maxLinearSamples,
    int maxAngularSamples, const CollisionChecker::ShapeType robotShapeType,
    const std::vector<float> robotDimensions,
    const Eigen::Vector3f &sensor_position_body,
    const Eigen::Quaternionf &sensor_rotation_body, const double octreeRes,
    const int maxNumThreads) {
  // Setup the collision checker
  collChecker = std::make_unique<CollisionChecker>(
      robotShapeType, robotDimensions, sensor_position_body,
      sensor_rotation_body, octreeRes);

  // Setup configuration parameters
  ctrlimits = controlLimits;
  ctrType = controlType;
  time_step_ = timeStep;
  max_time_ = predictionHorizon;
  base_max_time_ = predictionHorizon;
  control_time_ = controlHorizon;
  lin_samples_max_ = maxLinearSamples;
  computeLinearSampleSplit(ctrType, lin_samples_max_, lin_samples_x_,
                           lin_samples_y_);
  // make sure number of angular samples are odd so that zero is not skipped
  // when range is symmetrical around zero
  ang_samples_max_ = maxAngularSamples + 1 - (maxAngularSamples % 2);
  numPointsPerTrajectory = getNumPointsPerTrajectory(time_step_, max_time_);

  if (ctrType != ControlType::OMNI) {
    // Discard Vy limits to eliminate movement on Y axis
    ctrlimits.velYParams = LinearVelocityControlParams(0.0, 0.0, 0.0);
  }

  numTrajectories =
      getNumTrajectories(ctrType, lin_samples_max_, ang_samples_max_);

  this->maxNumThreads = maxNumThreads;
}

TrajectorySampler::TrajectorySampler(
    TrajectorySamplerParameters config, ControlLimitsParams controlLimits,
    ControlType controlType, const CollisionChecker::ShapeType robotShapeType,
    const std::vector<float> robotDimensions,
    const Eigen::Vector3f &sensor_position_body,
    const Eigen::Quaternionf &sensor_rotation_body, const int maxNumThreads) {
  double octreeRes = config.getParameter<double>("octree_map_resolution");
  // Setup the collision checker
  collChecker = std::make_unique<CollisionChecker>(
      robotShapeType, robotDimensions, sensor_position_body,
      sensor_rotation_body, octreeRes);
  // Setup configuration parameters
  ctrlimits = controlLimits;
  ctrType = controlType;

  updateParams(config);

  numPointsPerTrajectory = getNumPointsPerTrajectory(time_step_, max_time_);

  if (ctrType != ControlType::OMNI) {
    // Discard Vy limits to eliminate movement on Y axis
    ctrlimits.velYParams = LinearVelocityControlParams(0.0, 0.0, 0.0);
  }

  numTrajectories =
      getNumTrajectories(ctrType, lin_samples_max_, ang_samples_max_);
  numCtrlPoints_ = control_time_ / time_step_;

  this->maxNumThreads = maxNumThreads;
  this->drop_samples_ = config.getParameter<bool>("drop_samples");
}

void TrajectorySampler::resetOctreeResolution(const double resolution) {
  collChecker->resetOctreeResolution(resolution);
}

void TrajectorySampler::setSampleDroppingMode(const bool drop_samples) {
  this->drop_samples_ = drop_samples;
}

float TrajectorySampler::getRobotRadius() const {
  return collChecker->getRadius();
}

void TrajectorySampler::updateParams(TrajectorySamplerParameters config) {
  time_step_ = config.getParameter<double>("time_step");
  max_time_ = config.getParameter<double>("prediction_horizon");
  base_max_time_ = max_time_;
  control_time_ = config.getParameter<double>("control_horizon");
  lin_samples_max_ = config.getParameter<int>("max_linear_samples");
  computeLinearSampleSplit(ctrType, lin_samples_max_, lin_samples_x_,
                           lin_samples_y_);
  int maxAngularSamples = config.getParameter<int>("max_angular_samples");
  ang_samples_max_ = maxAngularSamples + 1 - (maxAngularSamples % 2);
}

void TrajectorySampler::getAdmissibleTrajsFromVel(
    const Velocity2D &vel, const Path::State &start_pose,
    TrajectorySamples2D *admissible_velocity_trajectories) {

  if (std::abs(vel.vx()) < MIN_VEL and std::abs(vel.vy()) < MIN_VEL and
      std::abs(vel.omega()) < MIN_VEL) {
    return;
  }
  Path::State simulated_pose = start_pose;
  TrajectoryVelocities2D simulated_velocities(numPointsPerTrajectory);
  TrajectoryPath path(numPointsPerTrajectory);
  int idx = 0;
  path.add(idx, start_pose.x, start_pose.y);
  bool is_collision = false;
  size_t last_free_index{numPointsPerTrajectory - 1};

  for (size_t i = 0; i < (numPointsPerTrajectory - 1); ++i) {
    simulated_pose.update(vel, time_step_);

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
      if (i > 0) {
        last_free_index = i - 1;
      }
      break;
    }
    simulated_velocities.add(i, vel);
    path.add(i + 1, simulated_pose.x, simulated_pose.y);
  }

  if (!drop_samples_ && is_collision && last_free_index > numCtrlPoints_ &&
      last_free_index < numPointsPerTrajectory - 1) {
    auto last_free_point = path.getIndex(last_free_index);
    for (size_t j = last_free_index + 1; j < (numPointsPerTrajectory - 1);
         ++j) {
      // Add zero vel
      simulated_velocities.add(j, Velocity2D(0.0, 0.0, 0.0));
      // Robot stays at the last simulated point
      path.add(j + 1, last_free_point.x(), last_free_point.y());
    }
    is_collision = false;
  }

  if (!is_collision) {
    if (maxNumThreads > 1) {
      std::lock_guard<std::mutex> lock(s_trajMutex);
      admissible_velocity_trajectories->push_back(simulated_velocities, path);
    } else {
      admissible_velocity_trajectories->push_back(simulated_velocities, path);
    }
  }
  return;
}

std::unique_ptr<TrajectorySamples2D>
TrajectorySampler::generateTrajectoriesNonHolonomic(
    const Velocity2D &current_vel, const Path::State &current_pose) {
  // create admissible trajectories container and pass raw ptr to generation
  // functions for the sake of multithreaded ownership handlings
  std::unique_ptr<TrajectorySamples2D> admissible_velocity_trajectories =
      std::make_unique<TrajectorySamples2D>(numTrajectories,
                                            numPointsPerTrajectory);
  // Sample the (vx × omega) grid for arc-like motion. Skip vx ≈ 0 so no
  // pure-rotation samples are produced — unexecutable for Ackermann, and
  // intentionally excluded for diff-drive (goal-cost cannot rank spins).
  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      if (std::abs(vx) >= MIN_VEL) {
        for (double omega = min_omega_; omega <= max_omega_;
             omega += ang_sample_resolution_) {

          Velocity2D vel = Velocity2D(vx, 0.0, omega); // Limit Y movement
          // Get admissible trajectories in separate threads
          pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this, vel,
                       current_pose, admissible_velocity_trajectories.get());
        }
      }
    }
  } else {
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      if (std::abs(vx) >= MIN_VEL) {
        for (double omega = min_omega_; omega <= max_omega_;
             omega += ang_sample_resolution_) {

          Velocity2D vel = Velocity2D(vx, 0.0, omega); // Limit Y movement
          getAdmissibleTrajsFromVel(vel, current_pose,
                                    admissible_velocity_trajectories.get());
        }
      }
    }
  }
  return admissible_velocity_trajectories;
}

std::unique_ptr<TrajectorySamples2D>
TrajectorySampler::generateTrajectoriesHolonomic(
    const Velocity2D &current_vel, const Path::State &current_pose) {
  std::unique_ptr<TrajectorySamples2D> admissible_velocity_trajectories =
      std::make_unique<TrajectorySamples2D>(numTrajectories,
                                            numPointsPerTrajectory);

  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    // vx, vy
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      for (double vy = min_vy_; vy <= max_vy_; vy += lin_sample_y_resolution_) {
        // discard rotation only movements
        if (std::abs(vx) < MIN_VEL && std::abs(vy) < MIN_VEL) {
          continue;
        }
        pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this,
                     Velocity2D(vx, vy, 0.0), current_pose,
                     admissible_velocity_trajectories.get());
      }
    }

    // vx, omega
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        // discard rotation only movements
        if (std::abs(vx) < MIN_VEL && std::abs(omega) < MIN_VEL) {
          continue;
        }
        pool.enqueue(&TrajectorySampler::getAdmissibleTrajsFromVel, this,
                     Velocity2D(vx, 0.0, omega), current_pose,
                     admissible_velocity_trajectories.get());
      }
    }
  } else {

    // vx, vy
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      for (double vy = min_vy_; vy <= max_vy_; vy += lin_sample_y_resolution_) {
        // discard rotation only movements
        if (std::abs(vx) < MIN_VEL && std::abs(vy) < MIN_VEL) {
          continue;
        }
        getAdmissibleTrajsFromVel(Velocity2D(vx, vy, 0.0), current_pose,
                                  admissible_velocity_trajectories.get());
      }
    }

    // vx, omega
    for (double vx = min_vx_; vx <= max_vx_; vx += lin_sample_x_resolution_) {
      for (double omega = min_omega_; omega <= max_omega_;
           omega += ang_sample_resolution_) {
        // discard rotation only movements
        if (std::abs(vx) < MIN_VEL && std::abs(omega) < MIN_VEL) {
          continue;
        }
        getAdmissibleTrajsFromVel(Velocity2D(vx, 0.0, omega), current_pose,
                                  admissible_velocity_trajectories.get());
      }
    }
  }
  return admissible_velocity_trajectories;
}

std::unique_ptr<TrajectorySamples2D>
TrajectorySampler::getNewTrajectories(const Velocity2D &current_vel,
                                      const Path::State &current_pose) {
  // Get the range of reachable velocities from the current velocity
  UpdateReachableVelocityRange(current_vel);

  switch (ctrType) {
  case ControlType::ACKERMANN:
    return generateTrajectoriesNonHolonomic(current_vel, current_pose);
  case ControlType::DIFFERENTIAL_DRIVE:
    return generateTrajectoriesNonHolonomic(current_vel, current_pose);
  case ControlType::OMNI:
    return generateTrajectoriesHolonomic(current_vel, current_pose);
  default:
    throw std::invalid_argument("Invalid control type");
  }
}

std::unique_ptr<TrajectorySamples2D>
TrajectorySampler::generateTrajectories(const Velocity2D &current_vel,
                                        const Path::State &current_pose,
                                        const Control::LaserScan &scan) {
  collChecker->updateState(current_pose);
  // Update the laserscan values at the current location -> no need to update
  // it as scan is not changing (during the simulation)
  collChecker->updateSensorData(scan);
  return getNewTrajectories(current_vel, current_pose);
}

std::unique_ptr<TrajectorySamples2D>
TrajectorySampler::generateTrajectories(const Velocity2D &current_vel,
                                        const Path::State &current_pose,
                                        const std::vector<Path::Point> &cloud) {
  collChecker->updateState(current_pose);
  // Update the PointCloud values
  collChecker->updateSensorData(cloud);
  return getNewTrajectories(current_vel, current_pose);
}

void TrajectorySampler::setPredictionHorizon(double horizon) {
  // Clamp to at least 2 time steps so rollout has room for a finite
  // difference, and never exceed the horizon passed at construction.
  const double min_horizon = 2.0 * time_step_;
  if (horizon < min_horizon)
    horizon = min_horizon;
  if (horizon > base_max_time_)
    horizon = base_max_time_;
  max_time_ = horizon;
  numPointsPerTrajectory = getNumPointsPerTrajectory(time_step_, max_time_);
}

void TrajectorySampler::UpdateReachableVelocityRange(
    Control::Velocity2D currentVel) {

  // The reachable velocity window is what the robot can physically change to
  // in the next *control* cycle.

  max_vx_ = std::min(ctrlimits.velXParams.maxVel,
                     currentVel.vx() +
                         ctrlimits.velXParams.maxAcceleration * time_step_);
  min_vx_ = std::max(-ctrlimits.velXParams.maxVel,
                     currentVel.vx() -
                         ctrlimits.velXParams.maxDeceleration * time_step_);

  if (ctrType == ControlType::OMNI) {
    max_vy_ = std::min(ctrlimits.velYParams.maxVel,
                       currentVel.vy() +
                           ctrlimits.velYParams.maxAcceleration * time_step_);
    min_vy_ = std::max(-ctrlimits.velYParams.maxVel,
                       currentVel.vy() -
                           ctrlimits.velYParams.maxDeceleration * time_step_);
  } else {
    max_vy_ = 0.0;
    min_vy_ = 0.0;
  }

  // Step cannot be zero -> creates infinite loop. Split samples between vx
  // and vy for omni so (vx_n * vy_n * omega_n) stays comparable to the
  // non-holonomic sample count.
  lin_sample_x_resolution_ =
      std::max((max_vx_ - min_vx_) / (lin_samples_x_ - 1), 0.001);
  lin_sample_y_resolution_ =
      (lin_samples_y_ > 1)
          ? std::max((max_vy_ - min_vy_) / (lin_samples_y_ - 1), 0.001)
          : 0.001;

  max_omega_ = std::min(ctrlimits.omegaParams.maxOmega,
                        currentVel.omega() +
                            ctrlimits.omegaParams.maxAcceleration * time_step_);
  min_omega_ = std::max(-ctrlimits.omegaParams.maxOmega,
                        currentVel.omega() -
                            ctrlimits.omegaParams.maxDeceleration * time_step_);

  ang_sample_resolution_ =
      std::max((max_omega_ - min_omega_) / (ang_samples_max_ - 1), 0.001);
}

void TrajectorySampler::updateState(const Path::State &current_state) {
  collChecker->updateState(current_state);
}

template <>
bool TrajectorySampler::checkStatesFeasibility<LaserScan>(
    const std::vector<Path::State> &states, const LaserScan &scan) {
  // collChecker->updateState(states[0]);
  collChecker->updateSensorData(scan);
  for (auto state : states) {
    collChecker->updateState(state);
    // Update the PointCloud values
    if (collChecker->checkCollisions()) {
      return true;
    }
  }
  return false;
}

template <>
bool TrajectorySampler::checkStatesFeasibility<std::vector<Path::Point>>(
    const std::vector<Path::State> &states,
    const std::vector<Path::Point> &cloud) {
  // collChecker->updateState(states[0]);
  collChecker->updateSensorData(cloud);
  for (auto state : states) {
    collChecker->updateState(state);
    // Update the PointCloud values
    if (collChecker->checkCollisions()) {
      return true;
    }
  }
  return false;
}

Trajectory2D
TrajectorySampler::generateSingleSampleFromVel(const Velocity2D &vel,
                                               const Path::State &pose) {
  Path::State simulated_pose = pose;
  Trajectory2D trajectory(numPointsPerTrajectory);
  trajectory.path.add(0, simulated_pose.x, simulated_pose.y);
  if (ctrType == ControlType::DIFFERENTIAL_DRIVE) {
    for (size_t i = 0; i < (numPointsPerTrajectory - 1); ++i) {
      if (std::abs(vel.vx()) > MIN_VEL && std::abs(vel.omega()) > MIN_VEL) {
        // Rotate then move
        Velocity2D tempVel = vel;
        tempVel.setVx(0.0);
        simulated_pose.update(tempVel, time_step_);
        trajectory.path.add(i + 1, simulated_pose.x, simulated_pose.y);
        trajectory.velocities.add(i, vel);

        tempVel.setVx(vel.vx());
        tempVel.setOmega(0.0);
        simulated_pose.update(tempVel, time_step_);
        trajectory.path.add(i + 1, simulated_pose.x, simulated_pose.y);
        trajectory.velocities.add(i, vel);
        i++;
      }
      // Else apply directly
      simulated_pose.update(vel, time_step_);
      trajectory.path.add(i + 1, simulated_pose.x, simulated_pose.y);
      trajectory.velocities.add(i, vel);
    }
  } else {
    for (size_t i = 0; i < (numPointsPerTrajectory - 1); ++i) {
      simulated_pose.update(vel, time_step_);
      trajectory.path.add(i + 1, simulated_pose.x, simulated_pose.y);
      trajectory.velocities.add(i, vel);
    }
  }
  return trajectory;
}

}; // namespace Control
} // namespace Kompass
