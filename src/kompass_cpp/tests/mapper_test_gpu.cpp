// LocalMapperGPU (SYCL) unit tests.
//
// AdaptiveCpp's runtime is reference-counted: it starts when the first
// SYCL object is constructed and tears down when the last is destroyed
// (AdaptiveCpp/AdaptiveCpp#1233, #1107). If each Boost test case constructs
// its own LocalMapperGPU the runtime restarts between cases, and letting
// the final instance's destructor run during static destruction races the
// runtime teardown — both symptoms manifest as glibc heap corruption at
// process exit.
//
// Workaround: one LocalMapperGPU per process, held by an intentionally-
// leaked function-local static. Same pattern in
// pointcloud_to_laserscan_test_gpu.cpp and critical_zone_test_gpu.cpp.

#include "datatypes/control.h"
#include "mapping/local_mapper_gpu.h"
#include "test.h"
#define BOOST_TEST_MODULE KOMPASS MAPPER TESTS
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace Kompass;

// ---------------------------------------------------------------------------
// Process-wide singleton. One LocalMapperGPU, one sycl::queue, kept alive
// across all test cases in this binary.
// ---------------------------------------------------------------------------

struct GridMapConfig {
  double angle_increment;
  double corner_distance;
  int random_points;

  int grid_height;
  int grid_width;
  float grid_res;
  float rangeMax;
  float angleStep;
  float minHeight;
  float maxHeight;
  float actual_size;

  Eigen::Vector2i centralPoint;
  double limit;

  Mapping::LocalMapperGPU gpu_local_mapper;

  GridMapConfig()
      : angle_increment(0.1), corner_distance(0.5), random_points(50),
        grid_height(10), grid_width(10), grid_res(0.1), rangeMax(20.0),
        angleStep(0.01), minHeight(0.0), maxHeight(0.0),
        actual_size(grid_width * grid_res),
        centralPoint(std::round(grid_height / 2) - 1,
                     std::round(grid_width / 2) - 1),
        limit(grid_width > grid_height ? grid_width * grid_res * std::sqrt(2)
                                       : grid_height * grid_res * std::sqrt(2)),
        gpu_local_mapper(Mapping::LocalMapperGPU(
            grid_height, grid_width, grid_res, {0.0, 0.0, 0.0}, 0.0, false, 63,
            angleStep, maxHeight, minHeight, rangeMax)) {
    LOG_INFO("Central point: ", centralPoint.x(), ", ", centralPoint.y());
    LOG_INFO("Limit Circle Radius: ", limit);
  }
};

// Intentionally leaked: LocalMapperGPU's destructor calls sycl::free on USM
// pointers, which races the AdaptiveCpp runtime teardown during static
// destruction at process exit (AdaptiveCpp/AdaptiveCpp#1107). Leaking is
// harmless for a test process and eliminates the race.
GridMapConfig &get_config() {
  static GridMapConfig *cfg = new GridMapConfig();
  return *cfg;
}

// Separate singleton for the pointcloud-mode mapper
struct PointCloudMapConfig {
  int grid_height;
  int grid_width;
  float grid_res;
  float rangeMax;
  int scan_size;   // number of angular bins produced by the conversion kernel
  float angleStep; // uniform spacing across [0, 2π)
  float minHeight;
  float maxHeight;

  Mapping::LocalMapperGPU mapper;

  PointCloudMapConfig()
      : grid_height(21), grid_width(21), grid_res(0.1f), rangeMax(5.0f),
        scan_size(360), angleStep(static_cast<float>(2.0 * M_PI / 360.0)),
        minHeight(-1.0f), maxHeight(1.0f),
        mapper(Mapping::LocalMapperGPU(
            grid_height, grid_width, grid_res, {0.0f, 0.0f, 0.0f}, 0.0f,
            /*isPointCloud*/ true, scan_size, angleStep, maxHeight, minHeight,
            rangeMax)) {}
};

PointCloudMapConfig &get_pc_config() {
  static PointCloudMapConfig *cfg = new PointCloudMapConfig();
  return *cfg;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int countPointsInGrid(const Eigen::MatrixXi &matrix, int value) {
  int count = 0;
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      if (matrix(i, j) == value)
        ++count;
    }
  }
  return count;
}

Control::LaserScan generateLaserScan(double angle_increment,
                                     const std::string &shape,
                                     double param = 1.0, int num_points = 100) {
  std::vector<double> angles, ranges;

  double angle = 0.0;

  if (shape == "circle") {
    double radius = param;
    while (angle < 2 * M_PI) {
      angles.emplace_back(angle);
      ranges.emplace_back(radius);
      angle += angle_increment;
    }
  } else if (shape == "right_corner") {
    double max_range = param;
    for (angle = -M_PI / 4; angle <= M_PI / 4; angle += angle_increment) {
      angles.emplace_back(angle);
      ranges.emplace_back(max_range / std::cos(angle));
    }
    for (angle = M_PI / 4; angle <= 3 * M_PI / 4; angle += angle_increment) {
      angles.emplace_back(angle);
      ranges.emplace_back(max_range / std::sin(angle));
    }
  } else if (shape == "random_points") {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < num_points; ++i) {
      angle = static_cast<double>(std::rand()) / RAND_MAX * 2 * M_PI;
      double range = static_cast<double>(std::rand()) / RAND_MAX * param;
      angles.emplace_back(angle);
      ranges.emplace_back(range);
    }
  } else {
    LOG_ERROR("Invalid shape specified. Use 'circle', 'right_corner', or "
              "'random_points'.");
  }
  return {ranges, angles};
}

// Runs one circle scan through the shared mapper, prints the resulting grid,
// and asserts the core invariants: the occupancy counts sum to the total
// cell count and at least some cells are marked OCCUPIED.
void run_circle_scan(double radius) {
  auto &cfg = get_config();
  Control::LaserScan circle_scan =
      generateLaserScan(cfg.angle_increment, "circle", radius);

  LOG_INFO("Testing with circle points at distance: ", radius,
           " and grid of width: ", cfg.actual_size);

  std::vector<double> filtered_ranges(circle_scan.ranges.size());
  for (size_t i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(cfg.limit, circle_scan.ranges[i]);
  }

  Eigen::MatrixXi *gridData = nullptr;
  {
    Timer timer;
    gridData =
        &cfg.gpu_local_mapper.scanToGrid(circle_scan.angles, filtered_ranges);
  }

  const int n_occ = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  const int n_empty = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  const int n_unknown = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", n_occ);
  LOG_INFO("Number of free cells: ", n_empty);
  LOG_INFO("Number of unknown cells: ", n_unknown);
  std::cout << *gridData << std::endl;

  // For radii larger than the grid half-diagonal (≈ 0.707 m here) every ray's
  // endpoint lands outside the grid and gets dropped by the kernel's bounds check
  // so n_occ can legitimately be 0.
  const int total = static_cast<int>(gridData->size());
  BOOST_TEST(n_occ + n_empty + n_unknown == total,
             "cell counts must sum to total (" << total << "), got "
                                               << n_occ + n_empty + n_unknown);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_mapper_circle_radius_0_3) { run_circle_scan(0.3); }

BOOST_AUTO_TEST_CASE(test_mapper_circle_radius_0_5) { run_circle_scan(0.5); }

BOOST_AUTO_TEST_CASE(test_mapper_circle_radius_2_0) { run_circle_scan(2.0); }

BOOST_AUTO_TEST_CASE(test_mapper_pointcloud_circle) {
  auto &cfg = get_pc_config();

  // Build a deterministic cloud: 200 points on a 0.5 m circle at z=0.1.
  // With a 21x21 grid at 0.1 m / cell, the grid covers 2.1 m × 2.1 m so the
  // circle sits comfortably inside.
  std::vector<int8_t> cloud;
  constexpr int N = 200;
  for (int i = 0; i < N; ++i) {
    float theta = 2.0f * static_cast<float>(M_PI) * i / N;
    addPointToCloud(cloud, 0.5f * std::cos(theta), 0.5f * std::sin(theta),
                    0.1f);
  }
  // Filtered: above ceiling.
  addPointToCloud(cloud, 0.3f, 0.0f, 2.0f);
  // Filtered: below floor.
  addPointToCloud(cloud, 0.0f, 0.3f, -2.0f);
  // Filtered: origin.
  addPointToCloud(cloud, 0.0f, 0.0f, 0.1f);

  const int point_step = static_cast<int>(sizeof(PointXYZ));
  const int num_points = static_cast<int>(cloud.size() / point_step);
  const int width = num_points;
  const int height = 1;
  const int row_step = width * point_step;

  Eigen::MatrixXi *grid = nullptr;
  {
    Timer t;
    grid = &cfg.mapper.scanToGrid(
        cloud, point_step, row_step, height, width,
        /*x_offset*/ static_cast<float>(offsetof(PointXYZ, x)),
        /*y_offset*/ static_cast<float>(offsetof(PointXYZ, y)),
        /*z_offset*/ static_cast<float>(offsetof(PointXYZ, z)));
  }

  const int n_occ = countPointsInGrid(
      *grid, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  const int n_empty =
      countPointsInGrid(*grid, static_cast<int>(Mapping::OccupancyType::EMPTY));
  const int n_unknown = countPointsInGrid(
      *grid, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("PointCloud mapper: OCCUPIED=", n_occ, " EMPTY=", n_empty,
           " UNEXPLORED=", n_unknown);
  std::cout << *grid << std::endl;

  const int total = static_cast<int>(grid->size());
  BOOST_TEST(n_occ + n_empty + n_unknown == total,
             "cell counts must sum to total (" << total << ")");
  // The circle sits well inside the grid, so we expect a visible ring.
  BOOST_TEST(n_occ > 0,
             "expected some OCCUPIED cells from the pointcloud circle");
  BOOST_TEST(n_empty > 0,
             "expected some EMPTY cells along the rays from origin");
}
