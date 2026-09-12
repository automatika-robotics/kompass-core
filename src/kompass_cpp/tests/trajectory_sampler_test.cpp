#include "controllers/controller.h"
#include "controllers/follower.h"
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "json_export.h"
#include "test.h"
#include "utils/logger.h"
#include "utils/trajectory_sampler.h"
#define BOOST_TEST_MODULE KOMPASS TRAJECTORY SAMPLER TESTS
#include <Eigen/Dense>
#include <algorithm>
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace Kompass;

// Generates the sample set of every robot type from one state, saves the
// samples and the reference path to JSON and plots them with
// trajectory_sampler_plt.py. The pictures are for inspection; the case fails
// when the plotting script fails.
BOOST_AUTO_TEST_CASE(plots_samples_for_each_robot_type) {

  Logger::getInstance().setLogLevel(LogLevel::DEBUG);

  // Create a test path
  // ------------------------------------------------------------------------------
  std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0),
                                  Path::Point(1.0, 0.0, 0.0),
                                  Path::Point(2.0, 0.0, 0.0)};
  Path::Path raw_path(points);

  // Generic follower to use for raw path interpolation and segmentation
  Control::Follower *follower = new Control::Follower();
  follower->setCurrentPath(raw_path);
  // Get the interpolated/segmented path
  Path::Path path = follower->getCurrentPath();

  // delete follower
  delete follower;
  // -----------------------------------------------------------------------------------------------------

  // Sampling configuration
  // ------------------------------------------------------------------------------
  double timeStep = 0.1;
  double predictionHorizon = 1.0;
  double controlHorizon = 0.2;
  int maxLinearSamples = 20;
  int maxAngularSamples = 20;
  int numThreads = 10;

  // Octomap resolution
  double octreeRes = 0.1;
  // -------------------------------------------------------------------------------------------------------

  // -------------------------------------------------------------------------------------------------------

  // Robot configuration
  // -----------------------------------------------------------------------------------
  Control::LinearVelocityControlParams x_params(1, 3, 5);
  Control::LinearVelocityControlParams y_params(1, 3, 5);
  Control::AngularVelocityControlParams angular_params(3.14, 3, 5, 8);
  Control::ControlLimitsParams controlLimits(x_params, y_params,
                                             angular_params);
  auto robotShapeType = Kompass::CollisionChecker::ShapeType::BOX;
  std::vector<float> robotDimensions{0.3, 0.3, 1.0};
  // std::array<float, 3> sensorPositionWRTbody {0.0, 0.0, 1.0};
  const Eigen::Vector3f sensor_position_body{0.0, 0.0, 0.5};
  const Eigen::Quaternionf sensor_rotation_body{0, 0, 0, 1};

  // Robot start state (pose)
  Path::State robotState(0.0, 0.0, 0.0, 0.0);

  // Robot laserscan value (empty)
  Control::LaserScan robotScan(toVecF({20.0, 10.0, 10.0}),
                               toVecF({0, 0.1, 0.2}));

  std::array<Control::ControlType, 3> robot_types = {
      Control::ControlType::ACKERMANN, Control::ControlType::DIFFERENTIAL_DRIVE,
      Control::ControlType::OMNI};

  // -------------------------------------------------------------------------------------------------------

  // RUN TEST FOR EACH ROBOT TYPE
  // ------------------------------------------------------------------------
  for (size_t j = 0; j < robot_types.size(); j++) {
    Control::TrajectorySampler trajSampler(
        controlLimits, robot_types[j], timeStep, predictionHorizon,
        controlHorizon, maxLinearSamples, maxAngularSamples, robotShapeType,
        robotDimensions, sensor_position_body, sensor_rotation_body, octreeRes,
        numThreads);

    // Robot initial velocity control
    Control::Velocity2D robotControl;
    std::unique_ptr<Control::TrajectorySamples2D> samples;

    LOG_INFO("TESTING ", Control::controlTypeToString(robot_types[j]));

    {
      Timer time;
      samples =
          trajSampler.generateTrajectories(robotControl, robotState, robotScan);
    }

    // Plot the trajectories (Save to json then run python script for plotting)
    boost::filesystem::path executablePath = boost::dll::program_location();
    std::string file_location = executablePath.parent_path().string();

    std::string trajectories_filename =
        file_location + "/trajectories_" +
        Control::controlTypeToString(robot_types[j]);
    std::string ref_path_filename = file_location + "/ref_path";

    saveTrajectoriesToJson(*samples, trajectories_filename + ".json");
    savePathToJson(path, ref_path_filename + ".json");

    std::string command =
        "python3 " + file_location + "/trajectory_sampler_plt.py --samples \"" +
        trajectories_filename + "\" --reference \"" + ref_path_filename + "\"";

    // Execute the Python script
    int res = system(command.c_str());
    BOOST_TEST(res == 0, "Plotting script failed with code " << res);
  }
}

namespace {

// A scan that sees nothing within 20 m in any direction
Control::LaserScan clearScan() {
  const int n = 36;
  Eigen::VectorXf ranges = Eigen::VectorXf::Constant(n, 20.0f);
  Eigen::VectorXf angles(n);
  for (int i = 0; i < n; ++i) {
    angles(i) = static_cast<float>(i) * 2.0f * M_PI / n;
  }
  return Control::LaserScan(ranges, angles);
}

// The sampler under test: 0.5 s steps like the Lite3 configuration, a
// cylinder of 0.2 m radius, the proximity sensor at the body origin. The
// creep speed comes from the minimum speed of the x-axis limits.
std::unique_ptr<Control::TrajectorySampler>
makeSampler(Control::ControlType type,
            const Control::LinearVelocityControlParams &x_params,
            int maxLinearSamples, int maxAngularSamples, int maxNumThreads,
            const Control::AngularVelocityControlParams &angular_params =
                Control::AngularVelocityControlParams(M_PI, 3.14, 2.0, 3.0)) {
  Control::LinearVelocityControlParams y_params(1.0, 5.0, 10.0, 0.0);
  Control::ControlLimitsParams limits(x_params, y_params, angular_params);
  return std::make_unique<Control::TrajectorySampler>(
      limits, type, 0.5, 5.0, 2.5, maxLinearSamples, maxAngularSamples,
      CollisionChecker::ShapeType::CYLINDER, std::vector<float>{0.2f, 0.4f},
      Eigen::Vector3f{0.0f, 0.0f, 0.0f}, Eigen::Quaternionf{1.0f, 0.0f, 0.0f, 0.0f},
      0.1, maxNumThreads);
}

// Number of samples whose first commanded vx equals the given value
size_t countVx(const Control::TrajectorySamples2D &samples, double vx) {
  size_t count = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (std::abs(samples.velocities.vx(i, 0) - vx) < 1e-6) {
      ++count;
    }
  }
  return count;
}

// Number of samples that command no motion at all (never expected)
size_t countStops(const Control::TrajectorySamples2D &samples) {
  size_t count = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (std::abs(samples.velocities.vx(i, 0)) < 1e-9 &&
        std::abs(samples.velocities.vy(i, 0)) < 1e-9 &&
        std::abs(samples.velocities.omega(i, 0)) < 1e-9) {
      ++count;
    }
  }
  return count;
}

// Number of samples that rotate in place (not allowed by the sampler)
size_t countPureSpins(const Control::TrajectorySamples2D &samples) {
  size_t count = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (std::abs(samples.velocities.vx(i, 0)) < 1e-9 &&
        std::abs(samples.velocities.vy(i, 0)) < 1e-9 &&
        std::abs(samples.velocities.omega(i, 0)) > 1e-9) {
      ++count;
    }
  }
  return count;
}

double smallestSpeed(const Control::TrajectorySamples2D &samples) {
  double smallest = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < samples.size(); ++i) {
    smallest = std::min(smallest, std::abs(static_cast<double>(
                                      samples.velocities.vx(i, 0))));
  }
  return smallest;
}

} // namespace

// At rest the reachable window is [-1, 1]: the grid of 5 speeds already
// holds 0 and +/-1, the creep speeds +/-0.05 are added with the full angular
// fan of a grid speed, and zero velocity is never sampled. Holds for every
// robot type, pooled and serial.
BOOST_AUTO_TEST_CASE(grid_holds_creep_speeds_for_every_robot_type) {
  const std::vector<Control::ControlType> types{
      Control::ControlType::ACKERMANN, Control::ControlType::DIFFERENTIAL_DRIVE,
      Control::ControlType::OMNI};
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.05);
  for (const auto type : types) {
    for (const int threads : {1, 4}) {
      auto sampler = makeSampler(type, x_params, 4, 4, threads);
      auto samples = sampler->generateTrajectories(
          Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());

      BOOST_TEST_CONTEXT("type " << Control::controlTypeToString(type)
                                 << ", threads " << threads) {
        BOOST_TEST(samples->size() <= sampler->numTrajectories);
        BOOST_TEST(countStops(*samples) == 0);
        BOOST_TEST(countPureSpins(*samples) == 0);
        BOOST_TEST(countVx(*samples, 0.05) >= 1);
        BOOST_TEST(countVx(*samples, -0.05) == countVx(*samples, 0.05));
        // The creep speed carries the same fan as a speed of the grid
        BOOST_TEST(countVx(*samples, 0.05) == countVx(*samples, 1.0));
        BOOST_TEST(countVx(*samples, 1.0) >= 1);
        BOOST_TEST(countVx(*samples, -1.0) >= 1);
      }
    }
  }
}

// Moving at 0.12 m/s with a window of +/-0.15 m/s around it, the grid of 7
// speeds holds neither 0 nor +/-0.05: 0 and +0.05 are inside the window and
// get added, -0.05 is outside and is not.
BOOST_AUTO_TEST_CASE(creep_speeds_are_added_only_inside_the_window) {
  const Control::LinearVelocityControlParams x_params(0.8, 0.3, 0.3, 0.05);
  auto sampler = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE, x_params,
                             6, 4, 1);
  auto samples = sampler->generateTrajectories(
      Control::Velocity2D(0.12, 0.0, 0.0), Path::State(0.0, 0.0, 0.0, 0.0),
      clearScan());

  BOOST_TEST(countStops(*samples) == 0);
  BOOST_TEST(countVx(*samples, 0.05) >= 1);
  BOOST_TEST(countVx(*samples, -0.05) == 0);
  BOOST_TEST(countVx(*samples, 0.07) == countVx(*samples, 0.05));
  // The creep speed is the slowest moving sample
  BOOST_TEST(std::abs(smallestSpeed(*samples) - 0.05) < 1e-6);
}

// A robot that cannot brake to zero within one step gets no creep speeds:
// every sample stays inside the window
BOOST_AUTO_TEST_CASE(no_creep_when_zero_is_not_reachable) {
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 0.2, 0.05);
  auto sampler = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE, x_params,
                             4, 4, 1);
  auto samples = sampler->generateTrajectories(
      Control::Velocity2D(1.0, 0.0, 0.0), Path::State(0.0, 0.0, 0.0, 0.0),
      clearScan());

  BOOST_TEST(samples->size() > 0);
  BOOST_TEST(countStops(*samples) == 0);
  BOOST_TEST(smallestSpeed(*samples) >= 0.9 - 1e-6);
}

// Number of samples whose first commanded omega equals the given value
size_t countOmega(const Control::TrajectorySamples2D &samples, double omega) {
  size_t count = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (std::abs(samples.velocities.omega(i, 0) - omega) < 1e-6) {
      ++count;
    }
  }
  return count;
}

// Grid values inside the robot's dead band are not sampled: with a minimum
// speed of 0.3 m/s the grid speeds +/-0.25 are gone and +/-0.3 take their
// place
BOOST_AUTO_TEST_CASE(grid_speeds_inside_the_dead_band_are_not_sampled) {
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.3);
  // 9 speeds over [-1, 1]: 0, +/-0.25, +/-0.5, +/-0.75, +/-1
  auto sampler = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE, x_params,
                             8, 4, 1);
  auto samples = sampler->generateTrajectories(
      Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());

  BOOST_TEST(countVx(*samples, 0.25) == 0);
  BOOST_TEST(countVx(*samples, -0.25) == 0);
  BOOST_TEST(countVx(*samples, 0.3) == countVx(*samples, 0.5));
  BOOST_TEST(countVx(*samples, -0.3) == countVx(*samples, 0.5));
  BOOST_TEST(countStops(*samples) == 0);
  for (size_t i = 0; i < samples->size(); ++i) {
    const double vx = samples->velocities.vx(i, 0);
    BOOST_TEST((vx == 0.0 || std::abs(vx) >= 0.3 - 1e-6));
  }
}

// The same for the angular axis: with a minimum rate of 0.4 rad/s the grid
// rates +/-0.3 are gone, +/-0.4 take their place, and straight driving
// (omega = 0) stays
BOOST_AUTO_TEST_CASE(angular_rates_inside_the_dead_band_are_not_sampled) {
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.05);
  // Window [-1.5, 1.5] in 11 slots: 0, +/-0.3, +/-0.6, ..., +/-1.5
  const Control::AngularVelocityControlParams angular_params(M_PI, 1.5, 3.0,
                                                             3.0, 0.4);
  auto sampler = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE, x_params,
                             4, 10, 1, angular_params);
  auto samples = sampler->generateTrajectories(
      Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());

  BOOST_TEST(countOmega(*samples, 0.3) == 0);
  BOOST_TEST(countOmega(*samples, -0.3) == 0);
  BOOST_TEST(countOmega(*samples, 0.4) >= 1);
  BOOST_TEST(countOmega(*samples, 0.4) == countOmega(*samples, 0.6));
  BOOST_TEST(countOmega(*samples, -0.4) == countOmega(*samples, 0.6));
  BOOST_TEST(countOmega(*samples, 0.0) >= 1);
  BOOST_TEST(countPureSpins(*samples) == 0);
  for (size_t i = 0; i < samples->size(); ++i) {
    const double omega = samples->velocities.omega(i, 0);
    BOOST_TEST((omega == 0.0 || std::abs(omega) >= 0.4 - 1e-6));
  }
}

// A creep speed that happens to be on the grid is not sampled twice
BOOST_AUTO_TEST_CASE(creep_speed_on_the_grid_is_not_duplicated) {
  const Control::LinearVelocityControlParams on_grid(1.0, 5.0, 10.0, 0.5);
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.05);
  auto with_creep = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE,
                                on_grid, 4, 4, 1);
  auto other = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE, x_params,
                           4, 4, 1);
  auto samples = with_creep->generateTrajectories(
      Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());
  auto reference = other->generateTrajectories(
      Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());

  BOOST_TEST(countVx(*samples, 0.5) == countVx(*reference, 0.5));
  BOOST_TEST(countVx(*samples, -0.5) == countVx(*reference, -0.5));
  BOOST_TEST(countVx(*samples, 0.05) == 0);
}

// A zero minimum speed means no creep samples
BOOST_AUTO_TEST_CASE(zero_minimum_speed_disables_the_creep_samples) {
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.0);
  auto sampler = makeSampler(Control::ControlType::DIFFERENTIAL_DRIVE, x_params,
                             4, 4, 1);
  auto samples = sampler->generateTrajectories(
      Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());

  BOOST_TEST(countStops(*samples) == 0);
  BOOST_TEST(countPureSpins(*samples) == 0);
  // Every moving sample is a grid speed (multiples of 0.5 m/s)
  for (size_t i = 0; i < samples->size(); ++i) {
    const double vx = samples->velocities.vx(i, 0);
    BOOST_TEST_CONTEXT("sample " << i << ", vx " << vx) {
      BOOST_TEST((vx == 0.0 || std::abs(std::abs(vx) - 0.5) < 1e-6 ||
                  std::abs(std::abs(vx) - 1.0) < 1e-6));
    }
  }
}

// Zero velocity is never a sample, at rest or in motion, in either sample
// dropping mode: a standing rollout would win in the middle of a route
BOOST_AUTO_TEST_CASE(zero_velocity_is_never_sampled) {
  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.05);
  for (const bool drop : {true, false}) {
    auto sampler = makeSampler(Control::ControlType::OMNI, x_params, 4, 4, 1);
    sampler->setSampleDroppingMode(drop);
    auto at_rest = sampler->generateTrajectories(
        Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());
    auto moving = sampler->generateTrajectories(
        Control::Velocity2D(0.5, 0.0, 0.0), Path::State(0.0, 0.0, 0.0, 0.0),
        clearScan());
    BOOST_TEST_CONTEXT("drop_samples " << drop) {
      BOOST_TEST(countStops(*at_rest) == 0);
      BOOST_TEST(countStops(*moving) == 0);
    }
  }
}

// The sample buffers are sized for the grid plus the three extra speeds and
// the two extra rates, for every robot type
BOOST_AUTO_TEST_CASE(capacity_counts_the_extra_samples) {
  // Differential drive, 4 and 4: (5 + 3) speeds x (5 + 2) rates
  BOOST_TEST(Control::getNumTrajectories(
                 Control::ControlType::DIFFERENTIAL_DRIVE, 4, 4) == 56u);
  // Omni, 20 and 20: (15 + 3) x (21 + 2) + (15 + 3) x (5 + 3)
  BOOST_TEST(Control::getNumTrajectories(Control::ControlType::OMNI, 20, 20) ==
             558u);

  const Control::LinearVelocityControlParams x_params(1.0, 5.0, 10.0, 0.05);
  const std::vector<Control::ControlType> types{
      Control::ControlType::ACKERMANN, Control::ControlType::DIFFERENTIAL_DRIVE,
      Control::ControlType::OMNI};
  for (const auto type : types) {
    auto sampler = makeSampler(type, x_params, 20, 20, 4);
    auto samples = sampler->generateTrajectories(
        Control::Velocity2D(), Path::State(0.0, 0.0, 0.0, 0.0), clearScan());
    BOOST_TEST_CONTEXT("type " << Control::controlTypeToString(type)) {
      BOOST_TEST(samples->size() <= sampler->numTrajectories);
      BOOST_TEST(countStops(*samples) == 0);
      BOOST_TEST(countVx(*samples, 0.05) >= 1);
      BOOST_TEST(countVx(*samples, -0.05) >= 1);
    }
  }
}
