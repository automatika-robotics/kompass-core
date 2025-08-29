#include "datatypes/control.h"
#include "mapping/local_mapper.h"
#include "test.h"
#define BOOST_TEST_MODULE KOMPASS MAPPER TESTS
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>
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
  float angleStep;
  float minHeight;
  float maxHeight;
  int maxNumThreads;

  double limit;
  int maxPointsPerLine;
  std::vector<double> filtered_ranges;

  Mapping::LocalMapper local_mapper;

  // Constructor to initialize the struct
  GridMapConfig()
      : angle_increment(0.1), corner_distance(0.5), random_points(50),
        grid_height(10), grid_width(10), grid_res(0.1),
        actual_size(grid_width * grid_res),
        centralPoint(std::round(grid_height / 2) - 1,
                     std::round(grid_width / 2) - 1),
        pPrior(0.6), pOccupied(0.9), pEmpty(1 - pOccupied), rangeSure(0.1),
        rangeMax(20.0), wallSize(0.2), angleStep(0.01), minHeight(0.0),
        maxHeight(0.0), maxNumThreads(10),
        limit(grid_width > grid_height ? grid_width * grid_res * std::sqrt(2)
                                       : grid_height * grid_res * std::sqrt(2)),
        maxPointsPerLine(static_cast<int>((limit / grid_res) * 1.5)),
        local_mapper(Mapping::LocalMapper(
            grid_height, grid_width, grid_res, {0.0, 0.0, 0.0}, 0.0, false, 0, pPrior,
            pOccupied, pEmpty, rangeSure, rangeMax, wallSize, angleStep,
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
  return {ranges, angles};
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
  for (size_t i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(limit, circle_scan.ranges[i]);
  }
  {

    Timer timer;
    auto [mat1, mat2] =
        local_mapper.scanToGridBaysian(circle_scan.angles, filtered_ranges);
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
  for (size_t i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(limit, circle_scan.ranges[i]);
  }
  {
    Timer timer;
    auto [mat1, mat2] =
        local_mapper.scanToGridBaysian(circle_scan.angles, filtered_ranges);
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
  for (size_t i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(limit, circle_scan.ranges[i]);
  }
  {
    Timer timer;
    local_mapper.scanToGridBaysian(circle_scan.angles, filtered_ranges);
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
