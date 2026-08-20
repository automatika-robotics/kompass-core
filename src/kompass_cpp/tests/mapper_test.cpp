#include "datatypes/control.h"
#include "mapping/local_mapper.h"
#include "test.h"
#define BOOST_TEST_MODULE KOMPASS MAPPER TESTS
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace Kompass;

struct GridMapConfig {
  double angle_increment;
  double corner_distance;
  int random_points;

  int grid_height;
  int grid_width;
  float grid_res;
  float actual_size;

  Eigen::Vector2i centralPoint;

  float pPrior;
  float pOccupied;
  float pEmpty;
  float rangeSure;
  float rangeMax;
  float wallSize;
  float minHeight;
  float maxHeight;
  int maxNumThreads;

  double limit;
  int maxPointsPerLine;
  Eigen::VectorXf filtered_ranges;

  Mapping::LocalMapper local_mapper;

  // Constructor to initialize the struct
  GridMapConfig()
      : angle_increment(0.1), corner_distance(0.5), random_points(50),
        grid_height(10), grid_width(10), grid_res(0.1),
        actual_size(grid_width * grid_res),
        centralPoint(std::round(grid_height / 2) - 1,
                     std::round(grid_width / 2) - 1),
        pPrior(0.6), pOccupied(0.9), pEmpty(1 - pOccupied), rangeSure(0.1),
        rangeMax(20.0), wallSize(0.2), minHeight(0.0),
        maxHeight(0.0), maxNumThreads(10),
        limit(grid_width > grid_height ? grid_width * grid_res * std::sqrt(2)
                                       : grid_height * grid_res * std::sqrt(2)),
        maxPointsPerLine(static_cast<int>((limit / grid_res) * 1.5)),
        local_mapper(Mapping::LocalMapper(
            grid_height, grid_width, grid_res, {SensorConfig{}}, false, 0,
            pPrior, pOccupied, pEmpty, rangeSure, rangeMax, wallSize,
            maxHeight, minHeight, maxPointsPerLine, maxNumThreads)) {

    // Logging the central point and limit circle radius
    // (for demonstration purposes)
    LOG_INFO("Central point: ", centralPoint.x(), ", ", centralPoint.y());
    LOG_INFO("Limit Circle Radius: ", limit);
    LOG_INFO("Max Steps in Grid inscribing the limit circle: ",
             maxPointsPerLine);
  }
};

int countPointsInGrid(const Eigen::MatrixXi &matrix, int value) {
  int count = 0;

  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      if (matrix(i, j) == value) {
        ++count;
      }
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
    // Generate points forming a circle of radius `param`
    double radius = param;
    while (angle < 2 * M_PI) {
      angles.emplace_back(angle);
      ranges.emplace_back(radius);
      angle += angle_increment;
    }
  } else if (shape == "right_corner") {
    // Generate points for a right corner
    double max_range = param;
    // Horizontal line
    for (angle = -M_PI / 4; angle <= M_PI / 4; angle += angle_increment) {
      angles.emplace_back(angle);
      ranges.emplace_back(max_range /
                          std::cos(angle)); // Distance along the horizontal
    }
    // Vertical line
    for (angle = M_PI / 4; angle <= 3 * M_PI / 4; angle += angle_increment) {
      angles.emplace_back(angle);
      ranges.emplace_back(max_range /
                          std::sin(angle)); // Distance along the vertical
    }
  } else if (shape == "random_points") {
    // Generate random points
    std::srand(
        static_cast<unsigned>(std::time(nullptr))); // Seed for randomness
    for (int i = 0; i < num_points; ++i) {
      angle = static_cast<double>(std::rand()) / RAND_MAX * 2 *
              M_PI; // Random angle
      double range = static_cast<double>(std::rand()) / RAND_MAX *
                     param; // Random range within max distance
      angles.emplace_back(angle);
      ranges.emplace_back(range);
    }
  } else {
    LOG_ERROR("Invalid shape specified. Use 'circle', 'right_corner', or "
              "'random_points'.");
  }
  return {toVecF(ranges), toVecF(angles)};
}

BOOST_FIXTURE_TEST_SUITE(s, GridMapConfig)

BOOST_AUTO_TEST_CASE(test_mapper_circles) {

  Eigen::MatrixXi *gridData = nullptr;
  Eigen::MatrixXf *gridDataProb = nullptr;
  // Generate circle scan with radius 0.3
  double radius = 0.3;
  Control::LaserScan circle_scan =
      generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           "and grid of width: ", actual_size);
  filtered_ranges.resize(circle_scan.ranges.size());
  for (Eigen::Index i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(static_cast<float>(limit), circle_scan.ranges[i]);
  }
  {

    Timer timer;
    auto [mat1, mat2] =
        local_mapper.scanToGridBayesian(circle_scan.angles, filtered_ranges);
    gridData = &mat1;
    gridDataProb = &mat2;
  }

  int occ_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  int free_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  int unknown_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  std::cout << *gridData << std::endl;
  std::cout << *gridDataProb << std::endl;

  // Generate circle scan with radius 0.5
  radius = 0.5; // Example radius for the circle
  circle_scan = generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           "and grid of width: ", actual_size);
  filtered_ranges.resize(circle_scan.ranges.size());
  for (Eigen::Index i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(static_cast<float>(limit), circle_scan.ranges[i]);
  }
  {
    Timer timer;
    auto [mat1, mat2] =
        local_mapper.scanToGridBayesian(circle_scan.angles, filtered_ranges);
    gridData = &mat1;
    gridDataProb = &mat2;
  }
  occ_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  free_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  unknown_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  std::cout << *gridData << std::endl;
  std::cout << *gridDataProb << std::endl;

  // Generate circle scan with radius 10.5
  radius = 2;
  circle_scan = generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           "and grid of width: ", actual_size);
  filtered_ranges.resize(circle_scan.ranges.size());
  for (Eigen::Index i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(static_cast<float>(limit), circle_scan.ranges[i]);
  }
  {
    Timer timer;
    local_mapper.scanToGridBayesian(circle_scan.angles, filtered_ranges);
  }
  occ_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  free_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  unknown_points = countPointsInGrid(
      *gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  std::cout << *gridData << std::endl;
  std::cout << *gridDataProb << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()

// ---------------------------------------------------------------------------
// Multi-sensor pointcloud fusion (CPU)
// ---------------------------------------------------------------------------

// Packs xyz points into a raw FLOAT32 buffer (point_step 12)
static std::vector<uint8_t> packXYZ(const std::vector<Eigen::Vector3f> &pts) {
  std::vector<uint8_t> data;
  data.reserve(pts.size() * 3 * sizeof(float));
  for (const auto &p : pts) {
    for (int k = 0; k < 3; ++k) {
      float v = p[k];
      const auto *bytes = reinterpret_cast<const uint8_t *>(&v);
      data.insert(data.end(), bytes, bytes + sizeof(float));
    }
  }
  return data;
}

static PointCloudView makeView(const std::vector<uint8_t> &data) {
  const int n = static_cast<int>(data.size() / 12);
  return PointCloudView{data, 12, 12 * n, 1, n, 0, 4, 8};
}

// Shared config for the multi-sensor cases: 20x20 grid at 0.1 m/cell
// (2 m x 2 m), 360 bins, body-frame band [0, 0.5]
static Mapping::LocalMapper
makeMultiMapper(const std::vector<SensorConfig> &sensors) {
  return Mapping::LocalMapper(20, 20, 0.1f, sensors, /*isPointCloud*/ true,
                              /*scanSize*/ 360, /*maxHeight*/ 0.5f,
                              /*minHeight*/ 0.0f, /*rangeMax*/ 5.0f,
                              /*maxPointsPerLine*/ 43, /*maxNumThreads*/ 1);
}

static int countOccupiedInRows(const Eigen::MatrixXi &grid, int rowBegin,
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

/**
 * Front + back mounted sensors, each seeing one obstacle straight ahead in
 * its own frame: the fused grid must contain occupied cells both ahead of
 * and behind the grid center, each roughly where the body-frame endpoint
 * lands, with free space carved from each sensor's own origin.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_fusion_front_back) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.3f, 0.0f, 0.2f}, 0.0f),
      SensorConfig::fromYaw({-0.3f, 0.0f, 0.2f}, static_cast<float>(M_PI))};
  auto mapper = makeMultiMapper(sensors);

  // One point 0.4 m straight ahead in each sensor's own frame, slightly
  // below its mount (body z = 0.1, inside the band)
  const auto cloudFront = packXYZ({{0.4f, 0.0f, -0.1f}}); // body (0.7, 0, 0.1)
  const auto cloudBack = packXYZ({{0.4f, 0.0f, -0.1f}});  // body (-0.7, 0, 0.1)

  const auto &grid =
      mapper.scanToGrid({makeView(cloudFront), makeView(cloudBack)});

  // Center row index is 9; the front obstacle lands around row 15-16, the
  // back one around row 2-4 (truncation quantizes the two sides
  // asymmetrically). Assert into halves with slack, and that each obstacle
  // sits near the center column
  BOOST_CHECK_GT(countOccupiedInRows(grid, 14, 17), 0); // ahead
  BOOST_CHECK_GT(countOccupiedInRows(grid, 1, 4), 0);   // behind
  // Both rays carved some free space
  BOOST_CHECK_GT(countPointsInGrid(
                     grid, static_cast<int>(Mapping::OccupancyType::EMPTY)),
                 0);
  LOG_INFO("Fused front+back grid (CPU):");
  std::cout << grid << std::endl;

  // Each cloud alone must leave the opposite half untouched
  const auto &gridFrontOnly =
      mapper.scanToGrid({makeView(cloudFront), PointCloudView{}});
  BOOST_CHECK_GT(countOccupiedInRows(gridFrontOnly, 14, 17), 0);
  BOOST_CHECK_EQUAL(countOccupiedInRows(gridFrontOnly, 0, 8), 0);
  LOG_INFO("Front-sensor-only grid (CPU):");
  std::cout << gridFrontOnly << std::endl;

  const auto &gridBackOnly =
      mapper.scanToGrid({PointCloudView{}, makeView(cloudBack)});
  BOOST_CHECK_GT(countOccupiedInRows(gridBackOnly, 1, 4), 0);
  BOOST_CHECK_EQUAL(countOccupiedInRows(gridBackOnly, 10, 19), 0);
  LOG_INFO("Back-sensor-only grid (CPU):");
  std::cout << gridBackOnly << std::endl;
}

/**
 * A yaw-rotated, offset mount must place the obstacle at the body-frame
 * location: sensor at (0.2, 0) yawed +90 deg seeing a point 0.5 m ahead in
 * its own frame -> body (0.2, 0.5), i.e. left of the robot, not ahead.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_mount_transform) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.2f, 0.0f, 0.0f}, static_cast<float>(M_PI_2))};
  auto mapper = makeMultiMapper(sensors);

  const auto cloud = packXYZ({{0.5f, 0.0f, 0.1f}}); // body (0.2, 0.5, 0.1)
  const auto &grid = mapper.scanToGrid({makeView(cloud)});

  // Expected cell around (11, 14); allow +-1 for truncation at cell borders
  bool found = false;
  for (int i = 10; i <= 12 && !found; ++i) {
    for (int j = 13; j <= 15 && !found; ++j) {
      found = grid(i, j) == static_cast<int>(Mapping::OccupancyType::OCCUPIED);
    }
  }
  BOOST_CHECK(found);
  // Nothing straight ahead of the robot (the naive un-transformed location)
  BOOST_CHECK_EQUAL(countOccupiedInRows(grid, 13, 19), 0);
  LOG_INFO("Yawed offset mount grid (CPU), obstacle expected left of center:");
  std::cout << grid << std::endl;
}

/**
 * Empty views mean "no data from this sensor": the other sensors still
 * contribute, and an all-empty batch yields an all-UNEXPLORED grid.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_empty_clouds) {
  const std::vector<SensorConfig> sensors = {
      SensorConfig::fromYaw({0.3f, 0.0f, 0.2f}, 0.0f),
      SensorConfig::fromYaw({-0.3f, 0.0f, 0.2f}, static_cast<float>(M_PI))};
  auto mapper = makeMultiMapper(sensors);

  const auto &gridAllEmpty =
      mapper.scanToGrid({PointCloudView{}, PointCloudView{}});
  BOOST_CHECK_EQUAL(
      countPointsInGrid(gridAllEmpty,
                        static_cast<int>(Mapping::OccupancyType::UNEXPLORED)),
      20 * 20);
}

/**
 * The single-cloud byte entry is an N=1 adapter onto the batched
 * path: identical output for the identical cloud.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_n1_equals_legacy_entry) {
  // A ring of points at 1 m, various heights inside the band
  std::vector<Eigen::Vector3f> pts;
  for (int d = 0; d < 360; d += 5) {
    const float a = d * static_cast<float>(M_PI) / 180.0f;
    pts.push_back({std::cos(a), std::sin(a), 0.1f + 0.001f * d});
  }
  const auto cloud = packXYZ(pts);

  // Identity mount, pointcloud mode
  Mapping::LocalMapper legacy(20, 20, 0.1f, {SensorConfig{}}, true, 360, 0.5f,
                              0.0f, 5.0f, 43, 1);
  const auto view = makeView(cloud);
  const Eigen::MatrixXi legacyGrid =
      legacy.scanToGrid(view.data, view.point_step, view.row_step, view.height,
                        view.width, view.x_offset, view.y_offset,
                        view.z_offset);

  auto multi = makeMultiMapper({SensorConfig{}}); // identity, N=1
  const auto &batchedGrid = multi.scanToGrid({view});

  BOOST_CHECK((legacyGrid.array() == batchedGrid.array()).all());
  BOOST_CHECK_GT(countPointsInGrid(batchedGrid,
                                   static_cast<int>(
                                       Mapping::OccupancyType::OCCUPIED)),
                 0);
}

/**
 * Error paths: cloud-count mismatch, negative offsets named per cloud,
 * batched call on a laserscan-configured mapper.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_error_paths) {
  auto mapper = makeMultiMapper(
      {SensorConfig{}, SensorConfig::fromYaw({-0.3f, 0.0f, 0.2f},
                                             static_cast<float>(M_PI))});

  const auto cloud = packXYZ({{1.0f, 0.0f, 0.1f}});

  // Wrong count
  BOOST_CHECK_THROW(mapper.scanToGrid({makeView(cloud)}),
                    std::invalid_argument);

  // Negative offset in the SECOND cloud: message must name clouds[1]
  PointCloudView bad = makeView(cloud);
  bad.y_offset = -4;
  BOOST_CHECK_EXCEPTION(
      mapper.scanToGrid({makeView(cloud), bad}), std::invalid_argument,
      [](const std::invalid_argument &e) {
        return std::string(e.what()).find("clouds[1]") != std::string::npos &&
               std::string(e.what()).find("non-negative") != std::string::npos;
      });

  // Laserscan-configured mapper cannot take clouds
  Mapping::LocalMapper scan_mapper(20, 20, 0.1f, {SensorConfig{}}, false, 0,
                                   0.5f, 0.0f, 5.0f, 43, 1);
  BOOST_CHECK_THROW(scan_mapper.scanToGrid({makeView(cloud)}),
                    std::logic_error);
}

// Packs xyz points into a raw FLOAT64 buffer (point_step 24, offsets 0/8/16)
static std::vector<uint8_t> packXYZ64(const std::vector<Eigen::Vector3f> &pts) {
  std::vector<uint8_t> data;
  data.reserve(pts.size() * 3 * sizeof(double));
  for (const auto &p : pts) {
    for (int k = 0; k < 3; ++k) {
      const double v = static_cast<double>(p[k]);
      const auto *bytes = reinterpret_cast<const uint8_t *>(&v);
      data.insert(data.end(), bytes, bytes + sizeof(double));
    }
  }
  return data;
}

/**
 * A FLOAT64 cloud must produce exactly the grid its FLOAT32 twin produces:
 * SensorConfig.cloud_field_type dispatches the CPU field decoding.
 */
BOOST_AUTO_TEST_CASE(test_multi_sensor_float64_cloud) {
  const std::vector<Eigen::Vector3f> points = {
      {0.4f, 0.0f, -0.1f}, {0.0f, 0.5f, -0.1f}, {-0.3f, -0.3f, -0.1f}};

  auto mapper32 = makeMultiMapper({SensorConfig::fromYaw({0.3f, 0.0f, 0.2f},
                                                         0.0f)});
  const auto cloud32 = packXYZ(points);
  const Eigen::MatrixXi grid32 = mapper32.scanToGrid({makeView(cloud32)});

  SensorConfig sensor64 = SensorConfig::fromYaw({0.3f, 0.0f, 0.2f}, 0.0f);
  sensor64.cloud_field_type = PointFieldType::FLOAT64;
  auto mapper64 = makeMultiMapper({sensor64});
  const auto cloud64 = packXYZ64(points);
  const int n = static_cast<int>(points.size());
  const PointCloudView view64{cloud64, 24, 24 * n, 1, n, 0, 8, 16};
  const Eigen::MatrixXi grid64 = mapper64.scanToGrid({view64});

  BOOST_CHECK_GT(countPointsInGrid(
                     grid32, static_cast<int>(Mapping::OccupancyType::OCCUPIED)),
                 0);
  BOOST_TEST((grid32.array() == grid64.array()).all(),
             "FLOAT64 cloud must produce the exact grid of its FLOAT32 twin");
}
