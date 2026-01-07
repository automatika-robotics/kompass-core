#include "benchmark_common.h"

// --- Project Includes ---
#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include "utils/logger.h"

// --- Cost Evaluator ---
#include "utils/cost_evaluator.h"

// --- Conditional Includes ---
#ifdef GPU
#include "mapping/local_mapper_gpu.h"
#include "utils/critical_zone_check_gpu.h"
#define PLATFORM_TAG "GPU_ACCELERATED"
#else
#include "mapping/local_mapper.h"
#include "utils/critical_zone_check.h"
#define PLATFORM_TAG "CPU_NATIVE"
#endif

#include <cstring> // for offsetof

using namespace Kompass;
using namespace Kompass::Benchmarks;

// =================================================================================
// 1. DATA GENERATORS
// =================================================================================

struct PointXYZ {
  float x, y, z, padding;
};

// Generates Heavy Trajectories for Cost Evaluator
std::unique_ptr<Control::TrajectorySamples2D>
generate_heavy_trajectory_samples(double predictionHorizon, double timeStep,
                                  int number_of_samples) {
  size_t number_of_points = static_cast<size_t>(predictionHorizon / timeStep);
  double v_1 = 1.0;
  double max_fluctuation = 0.5;

  auto samples = std::make_unique<Control::TrajectorySamples2D>(
      number_of_samples, number_of_points);

  // 1. Center Path
  Control::TrajectoryPath center_path(number_of_points);
  Control::TrajectoryVelocities2D center_vel(number_of_points);
  for (size_t i = 0; i < number_of_points; ++i) {
    center_path.add(i, Path::Point(timeStep * v_1 * i, 0.0, 0.0));
    if (i < number_of_points - 1)
      center_vel.add(i, Control::Velocity2D(v_1, 0.0, 0.0));
  }
  samples->push_back(center_vel, center_path);

  // 2. Generate Pairs
  int pairs = (number_of_samples - 1) / 2;
  double amp_step = max_fluctuation / (pairs > 0 ? pairs : 1);

  for (int p = 1; p <= pairs; ++p) {
    double current_amp = p * amp_step;

    // Linear Fluctuation
    Control::TrajectoryPath path_lin(number_of_points);
    Control::TrajectoryVelocities2D vel_lin(number_of_points);
    for (size_t i = 0; i < number_of_points; ++i) {
      double fluct_v = current_amp * std::sin(2 * M_PI * i / number_of_points);
      path_lin.add(
          i, Path::Point(timeStep * v_1 * i, timeStep * fluct_v * i, 0.0));
      if (i < number_of_points - 1)
        vel_lin.add(i, Control::Velocity2D(v_1, fluct_v, 0.0));
    }
    samples->push_back(vel_lin, path_lin);

    // Angular Fluctuation
    Control::TrajectoryPath path_ang(number_of_points);
    Control::TrajectoryVelocities2D vel_ang(number_of_points);
    for (size_t i = 0; i < number_of_points; ++i) {
      double fluct_ang =
          current_amp * std::cos(2 * M_PI * i / number_of_points);
      double x = timeStep * v_1 * i * std::cos(fluct_ang);
      double y = timeStep * v_1 * i * std::sin(fluct_ang);
      path_ang.add(i, Path::Point(x, y, 0.0));
      if (i < number_of_points - 1)
        vel_ang.add(i, Control::Velocity2D(v_1, 0.0, fluct_ang));
    }
    samples->push_back(vel_ang, path_ang);
  }
  return samples;
}

std::vector<int8_t> generate_heavy_pointcloud_bytes(size_t num_points) {
    // Create a temporary vector of strict types to ensure alignment
    std::vector<PointXYZ> temp_points(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        temp_points[i].x = (rand() % 2000) / 100.0f - 10.0f;
        temp_points[i].y = (rand() % 2000) / 100.0f - 10.0f;
        temp_points[i].z = (rand() % 300) / 100.0f;
        temp_points[i].padding = 0.0f;
    }

    // Memcpy into the byte buffer
    std::vector<int8_t> buffer(num_points * sizeof(PointXYZ));
    std::memcpy(buffer.data(), temp_points.data(), buffer.size());

    return buffer;
}

// Generates varied laserscan data for Mapping
void generate_mapping_scan(size_t num_points, std::vector<double> &ranges,
                           std::vector<double> &angles) {
  ranges.resize(num_points);
  angles.resize(num_points);
  double angle_step = (2.0 * M_PI) / num_points;
  for (size_t i = 0; i < num_points; ++i) {
    angles[i] = -M_PI + (i * angle_step);
    ranges[i] = 5.0 + 2.0 * std::sin(angles[i] * 20.0);
  }
}

// =================================================================================
// 2. MAIN RUNNER
// =================================================================================

int main(int argc, char *argv[]) {
  // 1. Setup Logging
  Kompass::setLogLevel(Kompass::LogLevel::INFO);

  if (argc < 3) {
    LOG_ERROR("Usage: ./kompass_benchmark <platform_name> <output_json_path>");
    return 1;
  }

  std::string platform_alias = argv[1];
  std::string output_path = argv[2];
  std::string log_file_path = output_path + ".log";
  Kompass::setLogFile(log_file_path);

  std::vector<BenchmarkResult> results;

  LOG_INFO("===================================================");
  LOG_INFO("  KOMPASS CORE BENCHMARK SUITE");
  LOG_INFO("  Mode:  ", PLATFORM_TAG);
  LOG_INFO("  Target:", platform_alias);
  LOG_INFO("===================================================");

  // -------------------------------------------------------------------------
  // TEST 1: COST EVALUATOR
  // -------------------------------------------------------------------------
  {
    double predictionHorizon = 10.0;
    double timeStep = 0.01;
    int numTrajectories = 5001;

    std::vector<Path::Point> points{Path::Point(0.0, 0.0, 0.0),
                                    Path::Point(5.0, 0.0, 0.0),
                                    Path::Point(10.0, 0.0, 0.0)};
    Path::Path reference_path(points);
    reference_path.interpolate(0.01, Path::InterpolationType::LINEAR);
    reference_path.segment(1000.0, 1000);

    auto samples = generate_heavy_trajectory_samples(predictionHorizon,
                                                     timeStep, numTrajectories);

    Control::LinearVelocityControlParams x_p(1, 3, 5), y_p(1, 3, 5);
    Control::AngularVelocityControlParams a_p(3.14, 3, 5, 8);
    Control::ControlLimitsParams limits(x_p, y_p, a_p);
    Control::CostEvaluator::TrajectoryCostsWeights weights;
    weights.setParameter("reference_path_distance_weight", 1.0);
    weights.setParameter("smoothness_weight", 1.0);
    weights.setParameter("jerk_weight", 1.0);
    weights.setParameter("goal_distance_weight", 1.0);

    Control::CostEvaluator costEval(weights, limits, numTrajectories,
                                    predictionHorizon / timeStep, 1000);

    auto workload = [&]() {
      costEval.getMinTrajectoryCost(samples, &reference_path,
                                    reference_path.getSegment(0));
    };

    results.push_back(measure_performance("CostEvaluator_5k_Trajs", workload));
  }

  // -------------------------------------------------------------------------
  // TEST 2: MAPPING
  // -------------------------------------------------------------------------
  {
    int height = 400;
    int width = 400;
    float res = 0.05f;
    std::vector<double> ranges, angles;
    generate_mapping_scan(3600, ranges, angles);

#ifdef GPU
    Mapping::LocalMapperGPU mapper(height, width, res, {0.0, 0.0, 0.0}, 0.0,
                                   false, 63, 0.01, 2.0, 0.0, 20.0);
    auto workload = [&]() { mapper.scanToGrid(angles, ranges); };
#else
    float limit = width * res * std::sqrt(2);
    int maxPointsPerLine = static_cast<int>((limit / res) * 1.5);
    Mapping::LocalMapper mapper(height, width, res, {0.0, 0.0, 0.0}, 0.0, false,
                                0, 0.6, 0.9, 0.1, 0.1, 20.0, 0.2, 0.01, 2.0,
                                0.0, maxPointsPerLine, 10);
    auto workload = [&]() { mapper.scanToGridBaysian(angles, ranges); };
#endif

    results.push_back(measure_performance("Mapper_Dense_400x400", workload));
  }

  // -------------------------------------------------------------------------
  // TEST 3: CRITICAL ZONE (Point Cloud)
  // -------------------------------------------------------------------------
  {
    auto shape = CollisionChecker::ShapeType::CYLINDER;
    std::vector<float> robotDim{0.51, 2.0};
    Eigen::Vector3f sensorPos{0.22, 0.0, 0.4};
    Eigen::Vector4f sensorRot{0, 0, 0.99, 0.0};
    float crit_angle = 160.0, crit_dist = 0.3, slow_dist = 0.6;
    std::vector<double> dummy_angles;
    dummy_angles.reserve(360);
    double angle_step = 2.0 * M_PI / 360.0;
    for (int i = 0; i < 360; ++i) {
      dummy_angles.push_back(i * angle_step);
    }
    auto cloud_bytes = generate_heavy_pointcloud_bytes(100000); // 100k points

    int point_step = sizeof(PointXYZ);
    int x_off = offsetof(PointXYZ, x);
    int y_off = offsetof(PointXYZ, y);
    int z_off = offsetof(PointXYZ, z);
    int num_points = cloud_bytes.size() / point_step;
    int width = num_points;
    int height = 1;
    int row_step = width * point_step;

#ifdef GPU
    CriticalZoneCheckerGPU checker(CriticalZoneChecker::InputType::POINTCLOUD,
                                   shape, robotDim, sensorPos, sensorRot,
                                   crit_angle, crit_dist, slow_dist,
                                   dummy_angles, 0.1, 2.0, 20.0);
#else
    CriticalZoneChecker checker(CriticalZoneChecker::InputType::POINTCLOUD,
                                shape, robotDim, sensorPos, sensorRot,
                                crit_angle, crit_dist, slow_dist, dummy_angles,
                                0.1, 2.0, 20.0);
#endif

    auto workload = [&]() {
      checker.check(cloud_bytes, point_step, row_step, height, width, x_off,
                    y_off, z_off, true);
    };

    results.push_back(measure_performance("CriticalZone_100k_Cloud", workload));
  }

  // -------------------------------------------------------------------------
  // TEST 4: CRITICAL ZONE (LaserScan)
  // -------------------------------------------------------------------------
  {
    // TARGET: Force points to be in the "Slowdown Zone" (0.81m to 1.11m from
    // center) This ensures the CPU loop does not break early (collision) and
    // does not skip (safe).
    // -----------------------------------------------------------------------
    // Robot Radius = 0.51
    // Critical Limit = 0.51 + 0.3 = 0.81
    // Slowdown Limit = 0.51 + 0.6 = 1.11
    // Target Distance = 0.96m (Safe middle ground)

    std::vector<double> ranges, angles;
    size_t num_scan_points = 3600;
    ranges.resize(num_scan_points);
    angles.resize(num_scan_points);
    double angle_step = (2.0 * M_PI) / num_scan_points;

    double sensor_x = 0.22; // From sensorPos below
    double target_dist_sq = 0.96 * 0.96;

    // Generate ranges such that every point lands exactly at 'target_dist' from
    // robot center
    for (size_t i = 0; i < num_scan_points; ++i) {
      angles[i] = -M_PI + (i * angle_step);

      // Equation for distance R from robot center given sensor offset sx:
      // R^2 = (sx + r*cos_theta)^2 + (r*sin_theta)^2
      // R^2 = sx^2 + 2*sx*r*cos_theta + r^2
      // 0 = r^2 + (2*sx*cos_theta)*r + (sx^2 - R^2)

      double b = 2.0 * sensor_x * std::cos(angles[i]);
      double c = (sensor_x * sensor_x) - target_dist_sq;

      // Quadratic formula: r = (-b + sqrt(b^2 - 4ac)) / 2a.  (a=1)
      double discriminant = b * b - 4.0 * c;
      if (discriminant >= 0) {
        ranges[i] = (-b + std::sqrt(discriminant)) / 2.0;
      } else {
        ranges[i] = 10.0; // Fallback (should not happen with these params)
      }
    }

    std::vector<float> robotDim{0.51, 2.0};
    Eigen::Vector3f sensorPos{0.22, 0.0, 0.4};
    Eigen::Vector4f sensorRot{0, 0, 0.99, 0.0};

#ifdef GPU
    CriticalZoneCheckerGPU checker(CriticalZoneChecker::InputType::LASERSCAN,
                                   CollisionChecker::ShapeType::CYLINDER,
                                   robotDim, sensorPos, sensorRot, 160.0, 0.3,
                                   0.6, angles, 0.1, 2.0, 20.0);
#else
    CriticalZoneChecker checker(CriticalZoneChecker::InputType::LASERSCAN,
                                CollisionChecker::ShapeType::CYLINDER, robotDim,
                                sensorPos, sensorRot, 160.0, 0.3, 0.6, angles,
                                0.1, 2.0, 20.0);
#endif

    auto workload = [&]() { checker.check(ranges, true); };

    results.push_back(measure_performance("CriticalZone_Dense_Scan", workload));
  }

  save_results_to_json(platform_alias, results, output_path);
  LOG_INFO("Benchmark suite completed.");

  return 0;
}
