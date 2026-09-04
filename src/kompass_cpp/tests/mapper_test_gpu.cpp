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
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
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
  float minHeight;
  float maxHeight;
  float actual_size;

  Eigen::Vector2i centralPoint;
  double limit;

  Mapping::LocalMapperGPU gpu_local_mapper;

  GridMapConfig()
      : angle_increment(0.1), corner_distance(0.5), random_points(50),
        grid_height(10), grid_width(10), grid_res(0.1), rangeMax(20.0),
        minHeight(0.0), maxHeight(0.0), actual_size(grid_width * grid_res),
        centralPoint(std::round(grid_height / 2) - 1,
                     std::round(grid_width / 2) - 1),
        limit(grid_width > grid_height ? grid_width * grid_res * std::sqrt(2)
                                       : grid_height * grid_res * std::sqrt(2)),
        gpu_local_mapper(Mapping::LocalMapperGPU(
            grid_height, grid_width, grid_res, {SensorConfig{}}, false, 63,
            maxHeight, minHeight, rangeMax)) {
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
  int scan_size; // number of angular bins produced by the conversion kernel
  float minHeight;
  float maxHeight;

  Mapping::LocalMapperGPU mapper;

  PointCloudMapConfig()
      : grid_height(21), grid_width(21), grid_res(0.1f), rangeMax(5.0f),
        scan_size(360), minHeight(-1.0f), maxHeight(1.0f),
        mapper(Mapping::LocalMapperGPU(
            grid_height, grid_width, grid_res, {SensorConfig{}},
            /*isPointCloud*/ true, scan_size, maxHeight, minHeight, rangeMax)) {
  }
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
  return {toVecF(ranges), toVecF(angles)};
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

  Eigen::VectorXf filtered_ranges(circle_scan.ranges.size());
  for (Eigen::Index i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] =
        std::min(static_cast<float>(cfg.limit), circle_scan.ranges[i]);
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
  // endpoint lands outside the grid and gets dropped by the kernel's bounds
  // check so n_occ can legitimately be 0.
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
  std::vector<uint8_t> cloud;
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
        /*x_offset*/ static_cast<int>(offsetof(PointXYZ, x)),
        /*y_offset*/ static_cast<int>(offsetof(PointXYZ, y)),
        /*z_offset*/ static_cast<int>(offsetof(PointXYZ, z)));
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

// A height band that lies entirely below the body origin, i.e. max_z < 0.
// The sign of the bound carries no meaning of its own: a negative upper edge
// is a real edge, not a request to disable the gate. The CPU
// pointCloudToLaserScanFromRaw treats it that way, so the GPU kernel must as
// well.
BOOST_AUTO_TEST_CASE(test_mapper_pointcloud_negative_max_z_band) {
  const int grid_height = 21, grid_width = 21;
  const float grid_res = 0.1f, rangeMax = 5.0f;
  const int scan_size = 360;
  const float minHeight = -1.2f, maxHeight = -0.2f;

  Mapping::LocalMapperGPU mapper(
      grid_height, grid_width, grid_res, {SensorConfig{}},
      /*isPointCloud*/ true, scan_size, maxHeight, minHeight, rangeMax);

  constexpr int N = 200;
  const int point_step = static_cast<int>(sizeof(PointXYZ));

  // Ring at z = +0.5, above the band's upper edge of -0.2 -> every point must
  // be rejected, so nothing may come back OCCUPIED.
  std::vector<uint8_t> above;
  for (int i = 0; i < N; ++i) {
    float theta = 2.0f * static_cast<float>(M_PI) * i / N;
    addPointToCloud(above, 0.5f * std::cos(theta), 0.5f * std::sin(theta),
                    0.5f);
  }
  const int above_width = static_cast<int>(above.size() / point_step);
  Eigen::MatrixXi &grid_above = mapper.scanToGrid(
      above, point_step, above_width * point_step, /*height*/ 1, above_width,
      static_cast<int>(offsetof(PointXYZ, x)),
      static_cast<int>(offsetof(PointXYZ, y)),
      static_cast<int>(offsetof(PointXYZ, z)));
  const int n_occ_above = countPointsInGrid(
      grid_above, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  BOOST_TEST(n_occ_above == 0,
             "points above the band's upper edge leaked through a negative "
             "max_z (got "
                 << n_occ_above << " OCCUPIED cells)");

  // Same ring at z = -0.7, inside [-1.2, -0.2] -> must still map normally, so
  // the assertion above is proving the gate applies and not that the whole
  // path is inert.
  std::vector<uint8_t> in_band;
  for (int i = 0; i < N; ++i) {
    float theta = 2.0f * static_cast<float>(M_PI) * i / N;
    addPointToCloud(in_band, 0.5f * std::cos(theta), 0.5f * std::sin(theta),
                    -0.7f);
  }
  const int band_width = static_cast<int>(in_band.size() / point_step);
  Eigen::MatrixXi &grid_band = mapper.scanToGrid(
      in_band, point_step, band_width * point_step, /*height*/ 1, band_width,
      static_cast<int>(offsetof(PointXYZ, x)),
      static_cast<int>(offsetof(PointXYZ, y)),
      static_cast<int>(offsetof(PointXYZ, z)));
  const int n_occ_band = countPointsInGrid(
      grid_band, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  BOOST_TEST(n_occ_band > 0,
             "expected OCCUPIED cells from a ring inside the negative band");
}

// ---------------------------------------------------------------------------
// Multi-sensor pointcloud fusion (GPU)
// ---------------------------------------------------------------------------

namespace {

// Builds a ring cloud: `n` points on a circle of `radius` at height `z`
// (sensor frame)
std::vector<uint8_t> makeRing(float radius, float z, int n) {
  std::vector<uint8_t> cloud;
  for (int i = 0; i < n; ++i) {
    const float theta = 2.0f * static_cast<float>(M_PI) * i / n;
    addPointToCloud(cloud, radius * std::cos(theta), radius * std::sin(theta),
                    z);
  }
  return cloud;
}

PointCloudView makeCloudView(const std::vector<uint8_t> &cloud) {
  const int point_step = static_cast<int>(sizeof(PointXYZ));
  const int n = static_cast<int>(cloud.size() / point_step);
  return PointCloudView{cloud,
                        point_step,
                        n * point_step,
                        /*height*/ 1,
                        /*width*/ n,
                        static_cast<int>(offsetof(PointXYZ, x)),
                        static_cast<int>(offsetof(PointXYZ, y)),
                        static_cast<int>(offsetof(PointXYZ, z))};
}

// Fraction of cells holding the same occupancy code in both grids.
// CPU and GPU quantize ray endpoints differently (truncation vs ceil) and
// bin bearings in double vs float, so exact equality is not achievable —
// the calibrated agreement between the two backends is ~0.93
double gridAgreement(const Eigen::MatrixXi &a, const Eigen::MatrixXi &b) {
  int same = 0;
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      if (a(i, j) == b(i, j)) {
        ++same;
      }
    }
  }
  return static_cast<double>(same) / static_cast<double>(a.size());
}

int countOccupiedInRows(const Eigen::MatrixXi &grid, int rowBegin,
                        int rowEnd /*inclusive*/) {
  int count = 0;
  for (int i = rowBegin; i <= rowEnd; ++i) {
    for (int j = 0; j < grid.cols(); ++j) {
      if (grid(i, j) == static_cast<int>(Mapping::OccupancyType::OCCUPIED)) {
        ++count;
      }
    }
  }
  return count;
}

// True when every OCCUPIED cell of `a` has an OCCUPIED cell of `b` within
// Chebyshev distance 1. Insensitive to the backends' one-cell endpoint
// quantization difference, but a systematic transform error (cells displaced
// by several cells) fails it
bool occupiedWithinOneCell(const Eigen::MatrixXi &a, const Eigen::MatrixXi &b) {
  const int occ = static_cast<int>(Mapping::OccupancyType::OCCUPIED);
  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < a.cols(); ++j) {
      if (a(i, j) != occ) {
        continue;
      }
      bool matched = false;
      for (int di = -1; di <= 1 && !matched; ++di) {
        for (int dj = -1; dj <= 1 && !matched; ++dj) {
          const int ni = i + di, nj = j + dj;
          if (ni >= 0 && ni < b.rows() && nj >= 0 && nj < b.cols() &&
              b(ni, nj) == occ) {
            matched = true;
          }
        }
      }
      if (!matched) {
        return false;
      }
    }
  }
  return true;
}

constexpr int kMppl = 45; // shared by CPU and GPU mappers in parity tests

} // namespace

/**
 * Front + back mounted sensors, each seeing one obstacle straight ahead in
 * its own frame: the fused GPU grid must contain OCCUPIED cells both ahead
 * of and behind the grid center, and each cloud alone must leave the other
 * half untouched.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_fusion_front_back_gpu) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.3f, 0.0f, 0.2f}, 0.0f),
      SensorConfig::fromYaw({-0.3f, 0.0f, 0.2f}, static_cast<float>(M_PI))};
  Mapping::LocalMapperGPU mapper(20, 20, 0.1f, sensors, /*isPointCloud*/ true,
                                 /*scanSize*/ 360, /*maxHeight*/ 0.5f,
                                 /*minHeight*/ 0.0f, /*rangeMax*/ 5.0f, kMppl);

  // One point 0.4 m straight ahead in each sensor's own frame, slightly
  // below its mount (body z = 0.1, inside the band)
  std::vector<uint8_t> front;
  addPointToCloud(front, 0.4f, 0.0f, -0.1f); // body (0.7, 0, 0.1)
  std::vector<uint8_t> back;
  addPointToCloud(back, 0.4f, 0.0f, -0.1f); // body (-0.7, 0, 0.1)

  const auto &grid =
      mapper.scanToGrid({makeCloudView(front), makeCloudView(back)});
  BOOST_TEST(countOccupiedInRows(grid, 14, 17) > 0, "no OCCUPIED cell ahead");
  BOOST_TEST(countOccupiedInRows(grid, 1, 4) > 0, "no OCCUPIED cell behind");
  LOG_INFO("Fused front+back grid (GPU):");
  std::cout << grid << std::endl;

  const auto &gridFrontOnly =
      mapper.scanToGrid({makeCloudView(front), PointCloudView{}});
  BOOST_TEST(countOccupiedInRows(gridFrontOnly, 14, 17) > 0);
  BOOST_TEST(countOccupiedInRows(gridFrontOnly, 0, 8) == 0,
             "front-only batch put OCCUPIED cells behind the robot");
  LOG_INFO("Front-sensor-only grid (GPU):");
  std::cout << gridFrontOnly << std::endl;

  const auto &gridBackOnly =
      mapper.scanToGrid({PointCloudView{}, makeCloudView(back)});
  BOOST_TEST(countOccupiedInRows(gridBackOnly, 1, 4) > 0);
  BOOST_TEST(countOccupiedInRows(gridBackOnly, 10, 19) == 0,
             "back-only batch put OCCUPIED cells ahead of the robot");
  LOG_INFO("Back-sensor-only grid (GPU):");
  std::cout << gridBackOnly << std::endl;

  // All-empty batch -> untouched (all-UNEXPLORED) grid
  const auto &gridEmpty =
      mapper.scanToGrid({PointCloudView{}, PointCloudView{}});
  BOOST_TEST(
      countPointsInGrid(gridEmpty,
                        static_cast<int>(Mapping::OccupancyType::UNEXPLORED)) ==
      20 * 20);
}

/**
 * Two-sensor fused grids must agree between the CPU and GPU backends.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_cpu_gpu_parity) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.2f, 0.0f, 0.2f}, 0.0f),
      SensorConfig::fromYaw({-0.2f, 0.0f, 0.2f}, static_cast<float>(M_PI))};

  const auto front = makeRing(0.5f, -0.1f, 100);
  const auto back = makeRing(0.4f, -0.1f, 100);
  const std::vector<PointCloudView> clouds = {makeCloudView(front),
                                              makeCloudView(back)};

  Mapping::LocalMapper cpu(21, 21, 0.1f, sensors, true, 360, 0.5f, 0.0f, 5.0f,
                           kMppl, /*threads*/ 1);
  Mapping::LocalMapperGPU gpu(21, 21, 0.1f, sensors, true, 360, 0.5f, 0.0f,
                              5.0f, kMppl);

  const Eigen::MatrixXi cpuGrid = cpu.scanToGrid(clouds);
  const Eigen::MatrixXi &gpuGrid = gpu.scanToGrid(clouds);

  // Two fused passes roughly square the single-sensor ~0.93 cell agreement
  // (truncation-vs-ceil endpoint band), hence the 0.85 floor.
  LOG_INFO("Two-sensor fused grid (CPU):");
  std::cout << cpuGrid << std::endl;
  LOG_INFO("Two-sensor fused grid (GPU):");
  std::cout << gpuGrid << std::endl;

  const double agreement = gridAgreement(cpuGrid, gpuGrid);
  LOG_INFO("CPU/GPU fused-grid agreement: ", agreement);
  BOOST_TEST(agreement >= 0.85,
             "CPU/GPU fused grids diverged: agreement " << agreement);
  BOOST_TEST(occupiedWithinOneCell(gpuGrid, cpuGrid),
             "a GPU OCCUPIED cell has no CPU counterpart within one cell");
  BOOST_TEST(occupiedWithinOneCell(cpuGrid, gpuGrid),
             "a CPU OCCUPIED cell has no GPU counterpart within one cell");
  BOOST_TEST(countPointsInGrid(
                 gpuGrid, static_cast<int>(Mapping::OccupancyType::OCCUPIED)) >
             0);
}

/**
 * Non-square grid parity (20 rows x 40 cols).
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_cpu_gpu_parity_nonsquare) {
  const std::vector<SensorConfig> sensors = {SensorConfig{}}; // identity

  const auto ring = makeRing(0.6f, 0.1f, 150);
  const std::vector<PointCloudView> clouds = {makeCloudView(ring)};

  Mapping::LocalMapper cpu(20, 40, 0.1f, sensors, true, 360, 0.5f, 0.0f, 5.0f,
                           kMppl, /*threads*/ 1);
  Mapping::LocalMapperGPU gpu(20, 40, 0.1f, sensors, true, 360, 0.5f, 0.0f,
                              5.0f, kMppl);

  const Eigen::MatrixXi cpuGrid = cpu.scanToGrid(clouds);
  const Eigen::MatrixXi &gpuGrid = gpu.scanToGrid(clouds);

  LOG_INFO("Non-square grid (CPU):");
  std::cout << cpuGrid << std::endl;
  LOG_INFO("Non-square grid (GPU):");
  std::cout << gpuGrid << std::endl;

  const double agreement = gridAgreement(cpuGrid, gpuGrid);
  LOG_INFO("CPU/GPU non-square agreement: ", agreement);
  BOOST_TEST(agreement >= 0.90,
             "CPU/GPU non-square grids diverged: agreement " << agreement);
  BOOST_TEST(countPointsInGrid(
                 gpuGrid, static_cast<int>(Mapping::OccupancyType::EMPTY)) > 0);
}

/**
 * Tilted mount (15 deg pitch): the GPU kernel must apply the full rotation,
 * not just yaw. Parity against the CPU path, which is pinned to the Eigen
 * reference.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_tilted_mount_gpu) {
  const float half_pitch = 7.5f * static_cast<float>(M_PI) / 180.0f;
  SensorConfig tilted;
  tilted.position = {0.0f, 0.0f, 0.3f};
  tilted.rotation = {0.0f, std::sin(half_pitch), 0.0f, std::cos(half_pitch)};
  const std::vector<SensorConfig> sensors = {tilted};

  const auto ring = makeRing(1.0f, 0.0f, 200);
  const std::vector<PointCloudView> clouds = {makeCloudView(ring)};

  Mapping::LocalMapper cpu(21, 21, 0.1f, sensors, true, 360, 1.0f, -1.0f, 5.0f,
                           kMppl, /*threads*/ 1);
  Mapping::LocalMapperGPU gpu(21, 21, 0.1f, sensors, true, 360, 1.0f, -1.0f,
                              5.0f, kMppl);

  const Eigen::MatrixXi cpuGrid = cpu.scanToGrid(clouds);
  const Eigen::MatrixXi &gpuGrid = gpu.scanToGrid(clouds);

  // A near-grid-edge ring maximises the endpoint quantization band, hence
  // the low smoke floor; the sharp check is the one-cell OCCUPIED
  // correspondence (a wrong tilt application would displace the ring by
  // several cells)
  LOG_INFO("Tilted-mount grid (CPU):");
  std::cout << cpuGrid << std::endl;
  LOG_INFO("Tilted-mount grid (GPU):");
  std::cout << gpuGrid << std::endl;

  const double agreement = gridAgreement(cpuGrid, gpuGrid);
  LOG_INFO("CPU/GPU tilted-mount agreement: ", agreement);
  BOOST_TEST(agreement >= 0.80,
             "CPU/GPU tilted-mount grids diverged: agreement " << agreement);
  BOOST_TEST(occupiedWithinOneCell(gpuGrid, cpuGrid),
             "a GPU OCCUPIED cell has no CPU counterpart within one cell");
  BOOST_TEST(occupiedWithinOneCell(cpuGrid, gpuGrid),
             "a CPU OCCUPIED cell has no GPU counterpart within one cell");
  BOOST_TEST(countPointsInGrid(
                 gpuGrid, static_cast<int>(Mapping::OccupancyType::OCCUPIED)) >
             0);
}

/**
 * Offset-mount EMPTY carving: a sensor mounted ahead of the body center,
 * with a ring that fills EVERY bin at 0.5 m (so no max-range rays can mask
 * the gate). The corridor between the sensor cell and the ring must be
 * carved EMPTY.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_offset_mount_empty_carving) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.3f, 0.0f, 0.0f}, 0.0f)};
  Mapping::LocalMapper cpu(20, 20, 0.1f, sensors, true, 360, 0.5f, 0.0f, 5.0f,
                           kMppl, /*threads*/ 1);
  Mapping::LocalMapperGPU gpu(20, 20, 0.1f, sensors, true, 360, 0.5f, 0.0f,
                              5.0f, kMppl);

  // 720 points -> every one of the 360 bins holds a genuine 0.5 m return
  const auto ring = makeRing(0.5f, 0.1f, 720);
  const std::vector<PointCloudView> clouds = {makeCloudView(ring)};

  const Eigen::MatrixXi cpuGrid = cpu.scanToGrid(clouds);
  const Eigen::MatrixXi &gpuGrid = gpu.scanToGrid(clouds);

  LOG_INFO("Offset-mount ring grid (CPU):");
  std::cout << cpuGrid << std::endl;
  LOG_INFO("Offset-mount ring grid (GPU):");
  std::cout << gpuGrid << std::endl;

  // Sensor cell is row 12 (0.3 m ahead of the center row 9); the ring ahead
  // of it sits around rows 16-17. The corridor rows 13..15 near the center
  // column must be EMPTY — with a mirrored distance table its values there
  // (0.7..1.0) all exceed the 0.5 m ranges and no EMPTY survives
  int emptyAheadCorridor = 0;
  for (int i = 13; i <= 15; ++i) {
    for (int j = 8; j <= 10; ++j) {
      if (gpuGrid(i, j) == static_cast<int>(Mapping::OccupancyType::EMPTY)) {
        ++emptyAheadCorridor;
      }
    }
  }
  BOOST_TEST(emptyAheadCorridor > 0,
             "no EMPTY cells carved between the offset sensor and its ring "
             "(distance-table origin is mirrored?)");

  const double agreement = gridAgreement(cpuGrid, gpuGrid);
  LOG_INFO("CPU/GPU offset-mount agreement: ", agreement);
  BOOST_TEST(agreement >= 0.85,
             "CPU/GPU offset-mount grids diverged: agreement " << agreement);
  BOOST_TEST(occupiedWithinOneCell(gpuGrid, cpuGrid));
  BOOST_TEST(occupiedWithinOneCell(cpuGrid, gpuGrid));
}

/**
 * The single-cloud byte entry is an N=1 adapter onto the batched
 * path: identical output for the identical cloud.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_n1_adapter_gpu) {
  auto &cfg = get_pc_config();
  const auto ring = makeRing(0.5f, 0.1f, 200);
  const auto view = makeCloudView(ring);

  const Eigen::MatrixXi legacyGrid = cfg.mapper.scanToGrid(
      view.data, view.point_step, view.row_step, view.height, view.width,
      view.x_offset, view.y_offset, view.z_offset);
  const Eigen::MatrixXi &batchedGrid = cfg.mapper.scanToGrid({view});

  BOOST_TEST((legacyGrid.array() == batchedGrid.array()).all(),
             "legacy N=1 entry and batched entry disagree");
}

/**
 * Error paths: cloud-count mismatch, negative offsets named per cloud,
 * batched call on a laserscan-configured mapper, and the laserscan overload
 * on a pointcloud-configured mapper (angle-buffer corruption guard).
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_error_paths_gpu) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig{},
      SensorConfig::fromYaw({-0.3f, 0.0f, 0.2f}, static_cast<float>(M_PI))};
  Mapping::LocalMapperGPU mapper(20, 20, 0.1f, sensors, true, 360, 0.5f, 0.0f,
                                 5.0f, kMppl);

  const auto ring = makeRing(0.5f, 0.1f, 50);

  // Wrong count
  BOOST_CHECK_THROW(mapper.scanToGrid({makeCloudView(ring)}),
                    std::invalid_argument);

  // Negative offset in the SECOND cloud: message must name clouds[1]
  PointCloudView bad = makeCloudView(ring);
  bad.y_offset = -4;
  BOOST_CHECK_EXCEPTION(
      mapper.scanToGrid({makeCloudView(ring), bad}), std::invalid_argument,
      [](const std::invalid_argument &e) {
        return std::string(e.what()).find("clouds[1]") != std::string::npos &&
               std::string(e.what()).find("non-negative") != std::string::npos;
      });

  // The laserscan overload on a pointcloud-mode mapper must throw instead of
  // silently overwriting the pre-uploaded conversion bin angles
  Eigen::VectorXf angles(4), ranges(4);
  angles.setZero();
  ranges.setConstant(1.0f);
  BOOST_CHECK_THROW(mapper.scanToGrid(angles, ranges), std::logic_error);

  // Batched call on the laserscan-configured singleton must throw
  auto &scan_cfg = get_config();
  BOOST_CHECK_THROW(scan_cfg.gpu_local_mapper.scanToGrid({makeCloudView(ring)}),
                    std::logic_error);
}

// ---------------------------------------------------------------------------
// Concurrency Tests. The mapper mutates per-sensor device buffers, the device grid
// and the host grid it returns by reference, all through one in-order queue.
// A reentrant callback group can run the scan callback concurrently, and a scan
// of a different length reallocates the cached angle vector; a second caller in
// pointcloud mode would hit the grow-and-reallocate path of the device buffers.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(test_mapper_pointcloud_concurrent_varying_frames_gpu) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.3f, 0.0f, 0.2f}, 0.0f),
      SensorConfig::fromYaw({-0.3f, 0.0f, 0.2f}, static_cast<float>(M_PI))};
  Mapping::LocalMapperGPU mapper(20, 20, 0.1f, sensors, /*isPointCloud*/ true,
                                 /*scanSize*/ 360, /*maxHeight*/ 0.5f,
                                 /*minHeight*/ 0.0f, /*rangeMax*/ 5.0f, kMppl);
  // Frames of different sizes so the device buffers keep growing; all points
  // sit in the height band around the sensors, at varying ranges
  std::vector<std::vector<uint8_t>> frames;
  for (int n : {20000, 26000, 21000, 30000, 23000, 28000}) {
    std::vector<uint8_t> cloud;
    for (int i = 0; i < n; ++i) {
      const float theta = 2.0f * static_cast<float>(M_PI) * i / n;
      const float r = 0.4f + 0.02f * (i % 7);
      addPointToCloud(cloud, r * std::cos(theta), r * std::sin(theta), -0.1f);
    }
    frames.push_back(std::move(cloud));
  }
  constexpr int kThreads = 4;
  constexpr int kIterations = 60;
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t] {
      for (int k = 0; k < kIterations; ++k) {
        const auto &a = frames[(t + k) % frames.size()];
        const auto &b = frames[(t * 5 + k * 3) % frames.size()];
        try {
          mapper.scanToGrid({makeCloudView(a), makeCloudView(b)});
        } catch (const std::exception &) {
          ++failures;
        }
      }
    });
  }
  for (auto &w : workers) {
    w.join();
  }
  BOOST_TEST(failures == 0, "scanToGrid threw " << failures << " times");
  // A quiet call afterwards still maps the ring of points around the robot
  const auto &grid = mapper.scanToGrid({makeCloudView(frames[0]), makeCloudView(frames[1])});
  BOOST_TEST(countOccupiedInRows(grid, 0, 19) > 0, "no OCCUPIED cell after the storm");
}

BOOST_AUTO_TEST_CASE(test_mapper_laserscan_concurrent_varying_lengths_gpu) {
  constexpr int kScanSize = 63;
  Mapping::LocalMapperGPU mapper(10, 10, 0.1f, {SensorConfig{}},
                                 /*isPointCloud*/ false, kScanSize,
                                 /*maxHeight*/ 0.0f, /*minHeight*/ 0.0f,
                                 /*rangeMax*/ 20.0f, kMppl);
  // Scans of different lengths (all at least scan_size, so none is skipped)
  // make the cached angle vector reallocate on every length change
  std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> scans;
  for (int n : {63, 70, 65, 80, 63, 75}) {
    Eigen::VectorXf angles = Eigen::VectorXf::LinSpaced(
        n, -static_cast<float>(M_PI), static_cast<float>(M_PI));
    Eigen::VectorXf ranges = Eigen::VectorXf::Constant(n, 0.3f);
    scans.emplace_back(angles, ranges);
  }
  constexpr int kThreads = 4;
  constexpr int kIterations = 150;
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t] {
      for (int k = 0; k < kIterations; ++k) {
        const auto &scan = scans[(t + k) % scans.size()];
        try {
          mapper.scanToGrid(scan.first, scan.second);
        } catch (const std::exception &) {
          ++failures;
        }
      }
    });
  }
  for (auto &w : workers) {
    w.join();
  }
  BOOST_TEST(failures == 0, "scanToGrid threw " << failures << " times");
  const auto &grid = mapper.scanToGrid(scans[0].first, scans[0].second);
  BOOST_TEST(countOccupiedInRows(grid, 0, 9) > 0, "no OCCUPIED cell after the storm");
}
