#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "json_export.h"
#include "test.h"
#include "utils/collision_check.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <cstddef>
#include <memory>
#define BOOST_TEST_MODULE KOMPASS COSTS TESTS
#include <boost/dll/runtime_symbol_info.hpp> // for program_location
#include <boost/filesystem.hpp>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass;
using namespace Kompass::Control;

// Helper to generate a dense 360-point LaserScan for stress testing
LaserScan generate_dense_scan(size_t num_points = 360, double range = 5.0,
                              double gap_angle_rad = 0.5) {
  std::vector<double> ranges;
  std::vector<double> angles;
  ranges.reserve(num_points);
  angles.reserve(num_points);

  double angle_step = (2.0 * M_PI) / static_cast<double>(num_points);
  double half_gap = gap_angle_rad / 2.0;

  for (size_t i = 0; i < num_points; ++i) {
    double current_angle = i * angle_step;

    // The x-axis corresponds to angle 0.
    // Since we wrap 0 to 2*PI, the "gap" is at the start and end of the
    // interval. We ONLY add the obstacle if it is OUTSIDE this gap.
    if (current_angle > half_gap && current_angle < (2.0 * M_PI - half_gap)) {
      ranges.push_back(range);
      angles.push_back(current_angle);
    }
  }
  return LaserScan(ranges, angles);
}

Path::Path generate_points_from_scan(const LaserScan &scan) {
  std::vector<Path::Point> points;
  for (size_t i = 0; i < scan.ranges.size(); ++i) {
    double x = scan.ranges[i] * std::cos(scan.angles[i]);
    double y = scan.ranges[i] * std::sin(scan.angles[i]);
    points.emplace_back(x, y, 0.0);
  }
  return Path::Path(points, points.size() + 1); // +1 buffer
}

std::unique_ptr<TrajectorySamples2D>
generate_ref_path_test_samples(double predictionHorizon, double timeStep,
                               int number_of_samples) {
  size_t number_of_points = static_cast<size_t>(predictionHorizon / timeStep);
  float vel = 1.0f;

  // Define the maximum spread
  float max_angle = static_cast<float>(M_PI) / 3.0f;

  std::unique_ptr<TrajectorySamples2D> samples =
      std::make_unique<TrajectorySamples2D>(number_of_samples,
                                            number_of_points);

  // Add the Center Reference (Angle 0)
  TrajectoryPath center_path(number_of_points);
  TrajectoryVelocities2D center_vel(number_of_points);

  for (size_t i = 0; i < number_of_points; ++i) {
    center_path.add(i, Path::Point(timeStep * vel * i, 0.0, 0.0));
    if (i < number_of_points - 1) {
      center_vel.add(i, Velocity2D(vel, 0.0, 0.0));
    }
  }
  samples->push_back(center_vel, center_path);

  // Add Pairs (Positive and Negative)
  int pairs = (number_of_samples - 1) / 2;

  // Calculate step so that the last pair reaches exactly max_angle
  float ang_step = max_angle / static_cast<float>(pairs);

  for (int p = 1; p <= pairs; ++p) {
    float angle_magnitude = p * ang_step;

    // We generate two samples per iteration: one positive, one negative
    std::vector<float> current_angles = {angle_magnitude, -angle_magnitude};

    for (float ang : current_angles) {
      TrajectoryPath path(number_of_points);
      TrajectoryVelocities2D vel_profile(number_of_points);

      for (size_t i = 0; i < number_of_points; ++i) {
        // Geometric path for a straight ray at 'ang'
        double x = timeStep * vel * i * std::cos(ang);
        double y = timeStep * vel * i * std::sin(ang);

        path.add(i, Path::Point(x, y, 0.0));

        if (i < number_of_points - 1) {
          vel_profile.add(i, Velocity2D(vel, 0.0, ang));
        }
      }
      samples->push_back(vel_profile, path);
    }
  }

  return samples;
}

std::unique_ptr<TrajectorySamples2D>
generate_smoothness_test_samples(double predictionHorizon, double timeStep,
                                 int number_of_samples) {
  size_t number_of_points = static_cast<size_t>(predictionHorizon / timeStep);
  double v_1 = 1.0;

  // Define max fluctuation amplitude (e.g. 0.5 m/s or 0.5 rad/s)
  double max_fluctuation = 0.5;

  std::unique_ptr<TrajectorySamples2D> samples =
      std::make_unique<TrajectorySamples2D>(number_of_samples,
                                            number_of_points);

  // Add the Center Reference (Smooth)
  // Constant velocity, no fluctuations.
  TrajectoryPath center_path(number_of_points);
  TrajectoryVelocities2D center_vel(number_of_points);

  for (size_t i = 0; i < number_of_points; ++i) {
    center_path.add(i, Path::Point(timeStep * v_1 * i, 0.0, 0.0));
    if (i < number_of_points - 1) {
      center_vel.add(i, Velocity2D(v_1, 0.0, 0.0));
    }
  }
  samples->push_back(center_vel, center_path);

  // Add Pairs (Linear Fluctuation vs Angular Fluctuation)
  int pairs = (number_of_samples - 1) / 2;
  double amp_step = max_fluctuation / static_cast<double>(pairs);

  for (int p = 1; p <= pairs; ++p) {
    double current_amp = p * amp_step;

    // --- Sample A: Linear Velocity Fluctuation (Simulates lateral jerk) ---
    TrajectoryPath path_lin(number_of_points);
    TrajectoryVelocities2D vel_lin(number_of_points);

    for (size_t i = 0; i < number_of_points; ++i) {
      // Fluctuation in Vy (Linear)
      double fluct_v = current_amp * std::sin(2 * M_PI * i / number_of_points);

      // x = v * t, y = t * fluctuation
      path_lin.add(
          i, Path::Point(timeStep * v_1 * i, timeStep * fluct_v * i, 0.0));

      if (i < number_of_points - 1) {
        // v_x constant, v_y fluctuates, omega 0
        vel_lin.add(i, Velocity2D(v_1, fluct_v, 0.0));
      }
    }
    samples->push_back(vel_lin, path_lin);

    // --- Sample B: Angular Velocity Fluctuation (Simulates rotational jerk)
    // ---
    TrajectoryPath path_ang(number_of_points);
    TrajectoryVelocities2D vel_ang(number_of_points);

    for (size_t i = 0; i < number_of_points; ++i) {
      // Fluctuation in Omega
      double fluct_ang =
          current_amp * std::cos(2 * M_PI * i / number_of_points);

      // Calculate position based on fluctuating heading
      double x = timeStep * v_1 * i * std::cos(fluct_ang);
      double y = timeStep * v_1 * i * std::sin(fluct_ang);

      path_ang.add(i, Path::Point(x, y, 0.0));

      if (i < number_of_points - 1) {
        // v_x constant, v_y 0, omega fluctuates
        vel_ang.add(i, Velocity2D(v_1, 0.0, fluct_ang));
      }
    }
    samples->push_back(vel_ang, path_ang);
  }

  return samples;
}

Trajectory2D run_test(CostEvaluator &costEval, Path::Path &reference_path,
                      std::unique_ptr<TrajectorySamples2D> &samples,
                      const size_t current_segment_index,
                      const double reference_path_distance_weight,
                      const double goal_distance_weight,
                      const double obstacles_distance_weight,
                      const double smoothness_weight, const double jerk_weight,
                      const bool plot_results = false,
                      const bool add_obstacles = false) {

  // Cost weights
  CostEvaluator::TrajectoryCostsWeights costWeights;
  costWeights.setParameter("reference_path_distance_weight",
                           reference_path_distance_weight);
  costWeights.setParameter("obstacles_distance_weight",
                           obstacles_distance_weight);
  costWeights.setParameter("goal_distance_weight", goal_distance_weight);
  costWeights.setParameter("smoothness_weight", smoothness_weight);
  costWeights.setParameter("jerk_weight", jerk_weight);
  costEval.updateCostWeights(costWeights);

  // Path for saving files
  boost::filesystem::path executablePath = boost::dll::program_location();
  std::string file_location = executablePath.parent_path().string();
  std::string filename_base =
      "cost_eval_ref_cost_" +
      std::to_string(static_cast<int>(reference_path_distance_weight * 100)) +
      "_goal_cost_" +
      std::to_string(static_cast<int>(goal_distance_weight * 100)) +
      "_smooth_cost_" +
      std::to_string(static_cast<int>(smoothness_weight * 100)) +
      "_jerk_cost_" + std::to_string(static_cast<int>(jerk_weight * 100)) +
      "_obs_cost_" +
      std::to_string(static_cast<int>(obstacles_distance_weight * 100));

  std::string traj_file = file_location + "/" + filename_base + "_traj";
  std::string ref_file = file_location + "/" + filename_base + "_ref";
  std::string obs_file = file_location + "/" + filename_base + "_obs";

  if (add_obstacles) {
    // Generate a dense scan (360 points) to stress the GPU kernel
    // place obstacles at 5.0m. Trajs go up to ~10m
    LaserScan robotScan = generate_dense_scan(360, 5.0);
    float maxObstaclesDist = 10.0f;

    // This converts LaserScan (Polar) -> PointCloud (Cartesian)
    costEval.setPointScan(robotScan, Path::State(), maxObstaclesDist);
    if (plot_results) {
      Path::Path obsPathObj = generate_points_from_scan(robotScan); // +1 buffer
      // save obstacles for plotting
      savePathToJson(obsPathObj, obs_file + ".json");
    }
  }

  TrajSearchResult result = costEval.getMinTrajectoryCost(
      samples, &reference_path, reference_path.segments[current_segment_index]);

  BOOST_TEST(result.isTrajFound,
             "Minimum reference path cost trajectory is not found!");

  if (result.isTrajFound) {
    LOG_INFO("Cost Evaluator Returned a Minimum Cost Path With Cost: ",
             result.trajCost);
  } else {
    throw std::logic_error(
        "Did not find any valid trajectory, this should not happen.");
  }

  // PLOT THE RESULTING TRAJECTORIES
  if (plot_results) {
    // Save the results to json
    saveTrajectoriesToJson(*samples.get(), traj_file + ".json");
    savePathToJson(reference_path, ref_file + ".json");

    // Command for running the python plot script using the saved files
    std::string command;
    if (add_obstacles) {
      command = "python3 " + file_location +
                "/trajectory_sampler_plt.py --samples \"" + traj_file +
                "\" --reference \"" + ref_file + "\" --obstacles \"" +
                obs_file + "\"";
    } else {
      command = "python3 " + file_location +
                "/trajectory_sampler_plt.py --samples \"" + traj_file +
                "\" --reference \"" + ref_file + "\"";
    }

    // Run the command
    int res = system(command.c_str());
    if (res != 0)
      throw std::system_error(res, std::generic_category(),
                              "Python script failed with error code");
  }

  return result.trajectory;
}

bool check_sample_equal_result(TrajectoryPath sample_path,
                               TrajectoryPath traj_path) {
  if (sample_path.x.size() != traj_path.x.size()) {
    LOG_INFO("Sample path and trajectory path do not contain the same number "
             "of points");
    return false;
  }
  double EPSILON = 0.001;
  for (Eigen::Index i = 0; i < sample_path.x.size(); ++i) {
    if ((std::abs(sample_path.x[i] - traj_path.x[i]) > EPSILON) ||
        (std::abs(sample_path.y[i] - traj_path.y[i])) > EPSILON) {
      return false;
    }
  }
  return true;
}

struct TestConfig {
  std::vector<Path::Point> points;
  Path::Path reference_path;
  double max_path_length;
  double max_interpolation_point_dist;
  size_t current_segment_index;
  double timeStep;
  double predictionHorizon;
  int maxNumThreads;
  LinearVelocityControlParams x_params;
  LinearVelocityControlParams y_params;
  AngularVelocityControlParams angular_params;
  ControlLimitsParams controlLimits;
  CollisionChecker::ShapeType robotShapeType;
  std::vector<float> robotDimensions;
  Eigen::Vector3f sensor_position_body;
  Eigen::Quaternionf sensor_rotation_body;
  CostEvaluator::TrajectoryCostsWeights costWeights;
  int numTrajectories;
  CostEvaluator costEval;
  bool plotResults;

  TestConfig()
      : points{Path::Point(0.0, 0.0, 0.0), Path::Point(5.0, 0.0, 0.0),
               Path::Point(10.0, 0.0, 0.0)},
        reference_path(points, 500), max_path_length(10.0),
        max_interpolation_point_dist(0.01), current_segment_index(0),
        timeStep(0.01), predictionHorizon(10.0), maxNumThreads(10),
        x_params(1, 3, 5), y_params(1, 3, 5), angular_params(3.14, 3, 5, 8),
        controlLimits(x_params, y_params, angular_params),
        robotShapeType(CollisionChecker::ShapeType::BOX),
        robotDimensions{0.3, 0.3, 1.0}, sensor_position_body{0.0, 0.0, 0.5},
        sensor_rotation_body{0, 0, 0, 1}, costWeights(), numTrajectories(1001),
        costEval(costWeights, controlLimits, numTrajectories,
                 predictionHorizon / timeStep,
                 max_path_length / max_interpolation_point_dist) {
    reference_path.setMaxLength(max_path_length);
    reference_path.interpolate(max_interpolation_point_dist,
                               Path::InterpolationType::LINEAR);
    reference_path.segment(1.0);
    // NOTE: Change numTrajectories to a manageable size (e.g 11) when setting
    // this to true
    plotResults = false;
  }
};

BOOST_FIXTURE_TEST_SUITE(s, TestConfig)

BOOST_AUTO_TEST_CASE(test_all_costs) {
  {
    LOG_INFO("Running Reference Path Cost Test");
    auto samples = generate_ref_path_test_samples(predictionHorizon, timeStep,
                                                  numTrajectories);
    Timer t;
    Trajectory2D min_traj =
        run_test(costEval, reference_path, samples, current_segment_index, 1.0,
                 0.0, 0.0, 0.0, 0.0, plotResults);

    // Validation (sample 0 is the best)
    BOOST_TEST(
        check_sample_equal_result(samples->getIndex(0).path, min_traj.path),
        "Minimum reference path cost trajectory is found but not "
        "equal to the correct minimum!");
  }

  {
    LOG_INFO("Running Goal Cost Test");
    auto samples = generate_ref_path_test_samples(predictionHorizon, timeStep,
                                                  numTrajectories);
    Timer t;
    Trajectory2D min_traj =
        run_test(costEval, reference_path, samples, current_segment_index, 0.0,
                 1.0, 0.0, 0.0, 0.0, plotResults);
    BOOST_TEST(
        check_sample_equal_result(samples->getIndex(0).path, min_traj.path),
        "Minimum goal cost trajectory is found but not "
        "equal to the correct minimum!");
  }

  {
    LOG_INFO("Running Smoothness Cost Test");
    auto samples = generate_smoothness_test_samples(predictionHorizon, timeStep,
                                                    numTrajectories);
    Timer t;
    Trajectory2D min_traj =
        run_test(costEval, reference_path, samples, current_segment_index, 0.0,
                 0.0, 0.0, 1.0, 0.0, plotResults);
    BOOST_TEST(
        check_sample_equal_result(samples->getIndex(0).path, min_traj.path),
        "Minimum smoothness cost trajectory is found but not "
        "equal to the correct minimum!");
  }

  {
    LOG_INFO("Running Jerk Cost Test");
    auto samples = generate_smoothness_test_samples(predictionHorizon, timeStep,
                                                    numTrajectories);
    Timer t;
    Trajectory2D min_traj =
        run_test(costEval, reference_path, samples, current_segment_index, 0.0,
                 0.0, 0.0, 0.0, 1.0, plotResults);
    BOOST_TEST(
        check_sample_equal_result(samples->getIndex(0).path, min_traj.path),
        "Minimum jerk cost trajectory is found but not "
        "equal to the correct minimum!");
  }

  {
    LOG_INFO("Running Obstacles Cost Test");
    auto samples = generate_ref_path_test_samples(predictionHorizon, timeStep,
                                                  numTrajectories);
    Timer t;
    Trajectory2D min_traj =
        run_test(costEval, reference_path, samples, current_segment_index, 0.0,
                 0.0, 1.0, 0.0, 0.0, plotResults, true);
    BOOST_TEST(
        check_sample_equal_result(samples->getIndex(0).path, min_traj.path),
        "Minimum obstacle cost trajectory is found but not "
        "equal to the correct minimum!");
  }
}

BOOST_AUTO_TEST_SUITE_END()
