#pragma once

#include "collision_check.h"
#include "datatypes/control.h"
#include "datatypes/parameter.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <array>
#include <cmath>
#include <vector>

#ifndef MIN_VEL
#define MIN_VEL 0.01
#endif

namespace Kompass {

namespace Control {
class TrajectorySampler {
public:
  class TrajectorySamplerParameters : public Parameters {
  public:
    TrajectorySamplerParameters() : Parameters() {
      addParameter(
          "time_step",
          Parameter(
              0.1, 0.001, 1000.0,
              "Time step in the trajectory points/control generation [sec]"));
      addParameter("prediction_horizon",
                   Parameter(1.0, 0.001, 1000.0,
                             "Future time horizon for the trajectory sampling "
                             "prediction [sec]"));
      addParameter(
          "control_horizon",
          Parameter(1.0, 0.001, 1000.0,
                    "Future time horizon for applying the control [sec]"));
      addParameter(
          "max_linear_samples",
          Parameter(
              10, 1, 1000,
              "Maximum number of samples for the linear velocity controls"));
      addParameter(
          "max_angular_samples",
          Parameter(
              10, 1, 1000,
              "Maximum number of samples for the angular velocity controls"));
      addParameter("octree_map_resolution",
                   Parameter(0.1, 0.0, 1000.0,
                             "Resolution of the built-in Octree map used for "
                             "collision checkings [m]"));
    }
  };

  /**
   * @brief Construct a new Trajectory Sampler object
   *
   * @param controlLimits
   * @param controlType
   * @param timeStep
   * @param timeHorizon
   * @param linearSampleStep
   * @param angularSampleStep
   * @param robotShapeType
   * @param robotDimensions
   * @param sensorPositionWRTbody
   * @param octreeRes
   */
  TrajectorySampler(ControlLimitsParams controlLimits, ControlType controlType,
                    double timeStep, double predictionHorizon,
                    double controlHorizon, int maxLinearSamples,
                    int maxAngularSamples,
                    const CollisionChecker::ShapeType robotShapeType,
                    const std::vector<float> robotDimensions,
                    const std::array<float, 3> &sensor_position_body,
                    const std::array<float, 4> &sensor_rotation_body,
                    const double octreeRes, const int maxNumThreads = 1);

  TrajectorySampler(TrajectorySamplerParameters config,
                    ControlLimitsParams controlLimits, ControlType controlType,
                    const CollisionChecker::ShapeType robotShapeType,
                    const std::vector<float> robotDimensions,
                    const std::array<float, 3> &sensor_position_body,
                    const std::array<float, 4> &sensor_rotation_body,
                    const int maxNumThreads = 1);

  /**
   * @brief Destroy the Trajectory Sampler object
   *
   */
  ~TrajectorySampler();

  /**
   * @brief Generates a set of trajectory samples based on a dynamic window of
   * valid acceleration actions. The generator returns valid trajectories only,
   * i.e. trajectories not resulting in collisions with obstacles
   *
   * @param current_vel   Current Velocity of the robot
   * @param current_pose  Current Location of the robot
   * @param scan          Sensor data (Laserscan)
   * @return std::vector<Trajectory>
   */
  std::vector<Trajectory> generateTrajectories(const Velocity &current_vel,
                                               const Path::State &current_pose,
                                               const LaserScan &scan);

  std::vector<Trajectory>
  generateTrajectories(const Velocity &current_vel,
                       const Path::State &current_pose,
                       const std::vector<Point3D> &cloud);

protected:
  // Protected member variables
  ControlType ctrType;
  ControlLimitsParams ctrlimits;
  CollisionChecker *collChecker;
  int maxNumThreads;

private:
  double time_step_{0.0};
  double max_time_{0.0};
  double control_time_{0.0};
  int lin_samples_max_;
  int ang_samples_max_;
  double lin_sample_x_resolution_;
  double lin_sample_y_resolution_;
  double ang_sample_resolution_;
  double min_velocity_{0.01};
  double max_vx_;
  double min_vx_;
  double max_vy_;
  double min_vy_;
  double max_omega_;
  double min_omega_;

  /**
   * @brief Helper method to update the class private parameters from config
   *
   * @param config
   */
  void updateParams(TrajectorySamplerParameters config);

  /**
   * @brief Updates the range of valid velocity actions that can be reached from
   * a current velocity based on the acceleration limits
   *
   * @param currentVel
   */
  void UpdateReachableVelocityRange(Control::Velocity currentVel);

  std::vector<Trajectory> getNewTrajectories(const Velocity &current_vel,
                                             const Path::State &current_pose);

  /**
   * @brief Get the admissible constant velocity trajectories from a starting
   * velocity value
   *
   * @param vel
   * @param start_pose
   * @param scan
   * @param admissible_velocity_trajectories
   */
  void getAdmissibleTrajsFromVel(
      const Velocity &vel, const Path::State &start_pose,
      std::vector<Trajectory> *admissible_velocity_trajectories);

  void getAdmissibleTrajsFromVelDiffDrive(
      const Velocity &vel, const Path::State &start_pose,
      std::vector<Trajectory> *admissible_velocity_trajectories);

  /**
   * @brief Generate trajectory samples for an ACKERMANN motion model
   *
   * @param current_vel
   * @param current_pose
   * @param scan
   * @return std::vector<Trajectory>
   */
  std::vector<Trajectory>
  generateTrajectoriesAckermann(const Velocity &current_vel,
                                const Path::State &current_pose);

  /**
   * @brief Generate trajectory samples for a DIFFERENTIAL_DRIVE motion model
   *
   * @param current_vel
   * @param current_pose
   * @param scan
   * @return std::vector<Trajectory>
   */
  std::vector<Trajectory>
  generateTrajectoriesDiffDrive(const Velocity &current_vel,
                                const Path::State &current_pose);

  /**
   * @brief Generate trajectory samples for an OMNI motion model
   *
   * @param current_vel
   * @param current_pose
   * @param scan
   * @return std::vector<Trajectory>
   */
  std::vector<Trajectory>
  generateTrajectoriesOmni(const Velocity &current_vel,
                           const Path::State &current_pose);
};
}; // namespace Control
} // namespace Kompass
