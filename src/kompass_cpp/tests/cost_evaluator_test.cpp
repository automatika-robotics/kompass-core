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

// --- Test Runner Function (uses the global costEval instance via reference)
// ---
Trajectory2D run_test(CostEvaluator &costEval, Path::Path &reference_path,
                      std::unique_ptr<TrajectorySamples2D> &samples,
                      const size_t current_segment_index,
                      const double reference_path_distance_weight,
                      const double goal_distance_weight,
                      const double obstacles_distance_weight,
                      const double smoothness_weight, const double jerk_weight,
                      const bool add_obstacles = false) {

  // Cost weights specific to this run
  CostEvaluator::TrajectoryCostsWeights run_costWeights;
  run_costWeights.setParameter("reference_path_distance_weight",
                               reference_path_distance_weight);
  run_costWeights.setParameter("obstacles_distance_weight",
                               obstacles_distance_weight);
  run_costWeights.setParameter("goal_distance_weight", goal_distance_weight);
  run_costWeights.setParameter("smoothness_weight", smoothness_weight);
  run_costWeights.setParameter("jerk_weight", jerk_weight);
  costEval.updateCostWeights(
      run_costWeights); // Update the *single* evaluator instance

  if (add_obstacles) {
    LaserScan robotScan({10.0f, 0.5f, 0.5f},
                        {0.0f, (float)M_PI_4, (float)(2.0 * M_PI - M_PI_4)});
    float maxObstaclesDist = 10.0f;
    // Update the *single* evaluator instance's obstacle data
    costEval.setPointScan(robotScan, Path::State(), maxObstaclesDist);
  }

  TrajSearchResult result = costEval.getMinTrajectoryCost(
      samples, reference_path, reference_path.segments[current_segment_index],
      current_segment_index);

  BOOST_TEST(result.isTrajFound, "Minimum cost trajectory is not found!");

  if (result.isTrajFound) {
    LOG_INFO("Cost Evaluator Returned a Minimum Cost Path With Cost: ",
             result.trajCost);
  } else {
    throw std::logic_error(
        "Did not find any valid trajectory, this should not happen.");
  }
  return result.trajectory;
}

bool check_sample_equal_result(const TrajectoryPath &sample_path,
                               const TrajectoryPath &traj_path) {
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

// --- Global Fixture Definition ---
struct TestConfig {
  // Static pointer to the single instance
  static TestConfig *instance;

  // --- Members needed by tests ---
  std::vector<Path::Point> points;
  Path::Path reference_path;
  double max_path_length;
  double max_interpolation_point_dist;
  size_t current_segment_index;
  double timeStep;
  double predictionHorizon;
  LinearVelocityControlParams x_params;
  LinearVelocityControlParams y_params;
  AngularVelocityControlParams angular_params;
  ControlLimitsParams controlLimits;
  CollisionChecker::ShapeType robotShapeType;
  std::vector<float> robotDimensions;
  std::array<float, 3> sensor_position_body;
  std::array<float, 4> sensor_rotation_body;
  CostEvaluator::TrajectoryCostsWeights costWeights; // Holds current weights
  CostEvaluator costEval; // The cost evaluator instance

  // Constructor: Setup runs ONCE
  TestConfig()
      : points{Path::Point(0.0, 0.0, 0.0), Path::Point(1.0, 0.0, 0.0),
               Path::Point(2.0, 0.0, 0.0)},
        reference_path(points), max_path_length(10.0),
        max_interpolation_point_dist(0.01), current_segment_index(0),
        timeStep(0.1), predictionHorizon(1.0),
        x_params(1, 3, 5), y_params(1, 3, 5), angular_params(3.14, 3, 5, 8),
        controlLimits(x_params, y_params, angular_params),
        robotShapeType(CollisionChecker::ShapeType::BOX),
        robotDimensions{0.3, 0.3, 1.0}, sensor_position_body{0.0, 0.0, 0.5},
        sensor_rotation_body{0.0f, 0.0f, 0.0f, 1.0f}, // Use 0.0f for float
        costEval(costWeights, controlLimits, 10,
                 static_cast<size_t>(predictionHorizon / timeStep),
                 static_cast<size_t>(max_path_length /
                                     max_interpolation_point_dist)) {
    reference_path.setMaxLength(max_path_length);
    reference_path.interpolate(max_interpolation_point_dist,
                               Path::InterpolationType::LINEAR);
    reference_path.segment(1.0);
  }

  // Destructor: Teardown runs ONCE
  ~TestConfig() {
    LOG_INFO("--- Global TestConfig Destroyed ONCE ---");
    /*instance = nullptr;*/
  }

  // Prevent copying/moving
  TestConfig(const TestConfig &) = delete;
  TestConfig &operator=(const TestConfig &) = delete;
  TestConfig(TestConfig &&) = delete;
  TestConfig &operator=(TestConfig &&) = delete;
};

// Define the static member outside the class
/*TestConfig *TestConfig::instance = nullptr;*/

// Register TestConfig as a global fixture managed by Boost Test
/*BOOST_GLOBAL_FIXTURE(TestConfig);*/

// --- Test Suite Definition ---
BOOST_FIXTURE_TEST_SUITE(s, TestConfig)

// Test Case 1: Reference Path Cost
BOOST_AUTO_TEST_CASE(test_reference_path_cost) {
  LOG_INFO("Running Reference Path Cost Test");
  // Generate test trajectory samples
  std::unique_ptr<TrajectorySamples2D> samples =
      generate_ref_path_test_samples(predictionHorizon, timeStep);

  Trajectory2D minimum_cost_traj =
      run_test(costEval, reference_path, samples, current_segment_index,
               1.0,  // reference_path_distance_weight
               0.0,  // goal_distance_weight
               0.0,  // obstacles_distance_weight
               0.0,  // smoothness_weight
               0.0); // jerk_weight
  // In the generated samples the first sample contains the minimum cost path
  Trajectory2D expected_sample =
      samples->getIndex(0); // Assuming getIndex exists
  bool check =
      check_sample_equal_result(expected_sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Reference Path Cost Test: Minimum cost trajectory is "
                    "found but not equal to the expected minimum! Got: "
                        << minimum_cost_traj.path.x
                        << ", Expected: " << expected_sample.path.x);
  if (check) {
    LOG_INFO("Reference Path Cost Test Passed!");
  }
}

// Test Case 2: Goal Distance Cost
BOOST_AUTO_TEST_CASE(test_goal_cost) {
  LOG_INFO("Running Goal Cost Test");
  // Generate test trajectory samples
  std::unique_ptr<TrajectorySamples2D> samples =
      generate_ref_path_test_samples(predictionHorizon, timeStep);

  Trajectory2D minimum_cost_traj =
      run_test(costEval, reference_path, samples, current_segment_index,
               0.0,  // reference_path_distance_weight
               1.0,  // goal_distance_weight
               0.0,  // obstacles_distance_weight
               0.0,  // smoothness_weight
               0.0); // jerk_weight
  Trajectory2D expected_sample = samples->getIndex(
      0); // Assuming sample 0 is still the best for goal cost in this setup
  bool check =
      check_sample_equal_result(expected_sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Goal Cost Test: Minimum cost trajectory is found but not "
                    "equal to the expected minimum! Got: "
                        << minimum_cost_traj.path.x
                        << ", Expected: " << expected_sample.path.x);
  if (check) {
    LOG_INFO("Goal Cost Test Passed!");
  }
}

// Test Case 3: Smoothness Cost
BOOST_AUTO_TEST_CASE(test_smoothness_cost) {
  LOG_INFO("Running Smoothness Cost Test");
  // Generate test trajectory samples specifically for smoothness
  std::unique_ptr<TrajectorySamples2D> samples =
      generate_smoothness_test_samples(predictionHorizon, timeStep);

  Trajectory2D minimum_cost_traj =
      run_test(costEval, reference_path, samples, current_segment_index,
               0.0,  // reference_path_distance_weight
               0.0,  // goal_distance_weight
               0.0,  // obstacles_distance_weight
               1.0,  // smoothness_weight
               0.0); // jerk_weight
  // Sample 0 in generate_smoothness_test_samples is designed to be the
  // smoothest
  Trajectory2D expected_sample = samples->getIndex(0);
  bool check =
      check_sample_equal_result(expected_sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Smoothness Cost Test: Minimum cost trajectory is found "
                    "but not equal to the expected minimum! Got: "
                        << minimum_cost_traj.path.x
                        << ", Expected: " << expected_sample.path.x);
  if (check) {
    LOG_INFO("Smoothness Cost Test Passed!");
  }
}

// Test Case 4: Jerk Cost
BOOST_AUTO_TEST_CASE(test_jerk_cost) {
  LOG_INFO("Running Jerk Cost Test");
  // Generate test trajectory samples specifically for smoothness/jerk
  std::unique_ptr<TrajectorySamples2D> samples =
      generate_smoothness_test_samples(predictionHorizon, timeStep);

  Trajectory2D minimum_cost_traj =
      run_test(costEval, reference_path, samples, current_segment_index,
               0.0,  // reference_path_distance_weight
               0.0,  // goal_distance_weight
               0.0,  // obstacles_distance_weight
               0.0,  // smoothness_weight
               1.0); // jerk_weight
  // Sample 0 in generate_smoothness_test_samples is also expected to have the
  // lowest jerk
  Trajectory2D expected_sample = samples->getIndex(0);
  bool check =
      check_sample_equal_result(expected_sample.path, minimum_cost_traj.path);
  BOOST_TEST(check, "Jerk Cost Test: Minimum cost trajectory is found but not "
                    "equal to the expected minimum! Got: "
                        << minimum_cost_traj.path.x
                        << ", Expected: " << expected_sample.path.x);
  if (check) {
    LOG_INFO("Jerk Cost Test Passed!");
  }
}

// Test Case 5: Obstacles Distance Cost
BOOST_AUTO_TEST_CASE(test_obstacles_cost) {
  LOG_INFO("Running Obstacles Cost Test");
  // Generate reference path samples (obstacles are added in run_test)
  std::unique_ptr<TrajectorySamples2D> samples =
      generate_ref_path_test_samples(predictionHorizon, timeStep);

  Trajectory2D minimum_cost_traj =
      run_test(costEval, reference_path, samples, current_segment_index,
               0.0,   // reference_path_distance_weight
               0.0,   // goal_distance_weight
               1.0,   // obstacles_distance_weight
               0.0,   // smoothness_weight
               0.0,   // jerk_weight
               true); // Add obstacles
  // With the specific obstacle setup (0.5m at +/- 45deg), sample 0 (straight
  // path) might still be the best if it doesn't collide or comes closest
  // safely. This depends heavily on the collision checking logic and exact
  // obstacle placement. Assuming sample 0 remains the expected best for this
  // simple obstacle scenario.
  Trajectory2D expected_sample = samples->getIndex(0);
  bool check =
      check_sample_equal_result(expected_sample.path, minimum_cost_traj.path);
  // Note: The assertion message might need refinement if sample 0 is *not* the
  // expected best with obstacles.
  BOOST_TEST(check, "Obstacles Cost Test: Minimum cost trajectory is found but "
                    "not equal to the expected minimum! Got: "
                        << minimum_cost_traj.path.x
                        << ", Expected: " << expected_sample.path.x);
  if (check) {
    LOG_INFO("Obstacles Cost Test Passed!");
  }
}

BOOST_AUTO_TEST_SUITE_END()
