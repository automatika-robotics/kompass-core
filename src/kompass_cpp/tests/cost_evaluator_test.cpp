#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "test.h"
#include "utils/collision_check.h"
#include "utils/cost_evaluator.h"
#include "utils/logger.h"
#include <cstddef>
#include <memory>
#define BOOST_TEST_MODULE KOMPASS COSTS TESTS
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <vector>

using namespace Kompass;
using namespace Kompass::Control;

std::unique_ptr<TrajectorySamples2D>
generate_ref_path_test_samples(double predictionHorizon, double timeStep) {
  size_t number_of_points = predictionHorizon / timeStep;
  float vel_1 = 1.0, vel_2 = 0.5, ang = 1.0;

  TrajectoryPath sample1(number_of_points), sample2(number_of_points),
      sample3(number_of_points);
  TrajectoryVelocities2D sample1_vel(number_of_points),
      sample2_vel(number_of_points), sample3_vel(number_of_points);

  for (size_t i = 0; i < number_of_points; ++i) {
    // sample 1 along the x-axis
    sample1.add(i, Path::Point(timeStep * vel_1 * i, 0.0, 0.0));

    // sample 2: positive ang
    sample2.add(i,
                Path::Point(timeStep * vel_2 * i * std::cos(ang * timeStep * i),
                            timeStep * vel_2 * i * std::sin(ang * timeStep * i),
                            0.0));
    // sample 3: negative ang
    sample3.add(
        i,
        Path::Point(timeStep * vel_2 * i * std::cos(-ang * timeStep * i),
                    timeStep * vel_2 * i * std::sin(-ang * timeStep * i), 0.0));

    if (i < number_of_points - 1) {
      sample1_vel.add(i, Velocity2D(vel_1, 0.0, 0.0));
      sample2_vel.add(i, Velocity2D(vel_2, 0.0, ang));
      sample3_vel.add(i, Velocity2D(vel_2, 0.0, -ang));
    }
  }

  std::unique_ptr<TrajectorySamples2D> samples =
      std::make_unique<TrajectorySamples2D>(3, number_of_points);
  samples->push_back(sample1_vel, sample1);
  samples->push_back(sample2_vel, sample2);
  samples->push_back(sample3_vel, sample3);
  return samples;
}

std::unique_ptr<TrajectorySamples2D>
generate_smoothness_test_samples(double predictionHorizon, double timeStep) {
  size_t number_of_points = predictionHorizon / timeStep;
  double v_1 = 1.0, ang1 = 0.1;

  TrajectoryPath sample1(number_of_points), sample2(number_of_points),
      sample3(number_of_points);
  TrajectoryVelocities2D sample1_vel(number_of_points),
      sample2_vel(number_of_points), sample3_vel(number_of_points);

  for (size_t i = 0; i < number_of_points; ++i) {
    // Sample 1: Constant velocity v_1 and constant angular velocity ang1
    sample1.add(i,
                Path::Point(timeStep * v_1 * i * std::cos(ang1 * timeStep * i),
                            timeStep * v_1 * i * std::sin(ang1 * timeStep * i),
                            0.0));

    // Sample 2: Fluctuations in linear velocity
    double fluctuation_v2 =
        v_1 + 0.1 * std::sin(2 * M_PI * i / number_of_points);
    double x2 = timeStep * fluctuation_v2 * i;
    double y2 = 0.0;
    sample2.add(i, Path::Point(x2, y2, 0.0));

    // Sample 3: Fluctuations in angular velocity
    double fluctuation_ang3 =
        ang1 + 0.1 * std::cos(2 * M_PI * i / number_of_points);
    double x3 = timeStep * v_1 * i * std::cos(fluctuation_ang3 * timeStep * i);
    double y3 = timeStep * v_1 * i * std::sin(fluctuation_ang3 * timeStep * i);
    sample3.add(i, Path::Point(x3, y3, 0.0));

    if (i < number_of_points - 1) {
      sample1_vel.add(i, Velocity2D(v_1, 0.0, ang1));
      sample2_vel.add(i, Velocity2D(fluctuation_v2, 0.0, 0.0));
      sample3_vel.add(i, Velocity2D(v_1, 0.0, fluctuation_ang3));
    }
  }

  std::unique_ptr<TrajectorySamples2D> samples =
      std::make_unique<TrajectorySamples2D>(3, number_of_points);
  samples->push_back(sample1_vel, sample1);
  samples->push_back(sample2_vel, sample2);
  samples->push_back(sample3_vel, sample3);
  return samples;
}

Trajectory2D run_test(CostEvaluator &costEval, Path::Path &reference_path,
                      std::unique_ptr<TrajectorySamples2D> &samples,
                      const size_t current_segment_index,
                      const double reference_path_distance_weight,
                      const double goal_distance_weight,
                      const double obstacles_distance_weight,
                      const double smoothness_weight, const double jerk_weight,
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

  if (add_obstacles) {
    // Robot laserscan value (empty)
    LaserScan robotScan({10.0, 0.5, 0.5}, {0, M_PI_4, 2 * M_PI - M_PI_4});
    float maxObstaclesDist = 10.0;
    costEval.setPointScan(robotScan, Path::State(), maxObstaclesDist);
  }

  TrajSearchResult result =
      costEval.getMinTrajectoryCost(samples, &reference_path, reference_path.segments[current_segment_index]);

  BOOST_TEST(result.isTrajFound,
             "Minimum reference path cost trajectory is not found!");

  if (result.isTrajFound) {
    LOG_INFO("Cost Evaluator Returned a Minimum Cost Path With Cost: ",
             result.trajCost);
  } else {
    throw std::logic_error(
        "Did not find any valid trajectory, this should not happen.");
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
  CostEvaluator costEval;

  TestConfig()
      : points{Path::Point(0.0, 0.0, 0.0), Path::Point(1.0, 0.0, 0.0),
               Path::Point(2.0, 0.0, 0.0)},
        reference_path(points, 500), max_path_length(10.0),
        max_interpolation_point_dist(0.01), current_segment_index(0),
        timeStep(0.1), predictionHorizon(1.0), maxNumThreads(10),
        x_params(1, 3, 5), y_params(1, 3, 5), angular_params(3.14, 3, 5, 8),
        controlLimits(x_params, y_params, angular_params),
        robotShapeType(CollisionChecker::ShapeType::BOX),
        robotDimensions{0.3, 0.3, 1.0}, sensor_position_body{0.0, 0.0, 0.5},
        sensor_rotation_body{0, 0, 0, 1}, costWeights(),
        costEval(costWeights, controlLimits, 10, predictionHorizon / timeStep,
                 max_path_length / max_interpolation_point_dist) {
    reference_path.setMaxLength(max_path_length);
    reference_path.interpolate(max_interpolation_point_dist,
                               Path::InterpolationType::LINEAR);
    reference_path.segment(1.0);
  }
};

BOOST_FIXTURE_TEST_SUITE(s, TestConfig)

BOOST_AUTO_TEST_CASE(test_all_costs) {
  // Create timer
  Timer time;
  LOG_INFO("Running Reference Path Cost Test");
  // Generate test trajectory samples
  std::unique_ptr<TrajectorySamples2D> samples =
      generate_ref_path_test_samples(predictionHorizon, timeStep);

  Trajectory2D minimum_cost_traj = run_test(costEval, reference_path, samples,
                               current_segment_index, 1.0, 0.0, 0.0, 0.0, 0.0);
  // In the generated samples the first sample contains the minimum cost path
  Trajectory2D sample = samples->getIndex(0);
  bool check = check_sample_equal_result(sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Minimum reference path cost trajectory is found but not "
                    "equal to the correct minimum! "
                        << minimum_cost_traj.path.x);
  if (check) {
    LOG_INFO("Test Passed!");
  }

  LOG_INFO("Running Goal Cost Test");
  // Generate test trajectory samples
  samples = generate_ref_path_test_samples(predictionHorizon, timeStep);

  minimum_cost_traj = run_test(costEval, reference_path, samples,
                               current_segment_index, 0.0, 1.0, 0.0, 0.0, 0.0);
  sample = samples->getIndex(0);
  check = check_sample_equal_result(sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Minimum reference path cost trajectory is found but not "
                    "equal to the correct minimum! "
                        << minimum_cost_traj.path.x);
  if (check) {
    LOG_INFO("Test Passed!");
  }

  LOG_INFO("Running Smoothness Cost Test");
  // Generate test trajectory samples
  samples = generate_smoothness_test_samples(predictionHorizon, timeStep);

  minimum_cost_traj = run_test(costEval, reference_path, samples,
                               current_segment_index, 0.0, 0.0, 0.0, 1.0, 0.0);
  sample = samples->getIndex(0);
  check = check_sample_equal_result(sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Minimum reference path cost trajectory is found but not "
                    "equal to the correct minimum! "
                        << minimum_cost_traj.path.x);
  if (check) {
    LOG_INFO("Test Passed!");
  }

  LOG_INFO("Running Jerk Cost Test");
  // Generate test trajectory samples
  samples = generate_smoothness_test_samples(predictionHorizon, timeStep);

  minimum_cost_traj = run_test(costEval, reference_path, samples,
                               current_segment_index, 0.0, 0.0, 0.0, 0.0, 1.0);
  sample = samples->getIndex(0);
  check = check_sample_equal_result(sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Minimum reference path cost trajectory is found but not "
                    "equal to the correct minimum! "
                        << minimum_cost_traj.path.x);
  if (check) {
    LOG_INFO("Test Passed!");
  }

  LOG_INFO("Running Obstacles Cost Test");
  // Generate test trajectory samples
  samples = generate_ref_path_test_samples(predictionHorizon, timeStep);

  minimum_cost_traj =
      run_test(costEval, reference_path, samples, current_segment_index, 0.0,
               0.0, 1.0, 0.0, 0.0, true);
  sample = samples->getIndex(0);
  check = check_sample_equal_result(sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Minimum reference path cost trajectory is found but not "
                    "equal to the correct minimum! "
                        << minimum_cost_traj.path.x);
  if (check) {
    LOG_INFO("Test Passed!");
  }
}

BOOST_AUTO_TEST_SUITE_END()
