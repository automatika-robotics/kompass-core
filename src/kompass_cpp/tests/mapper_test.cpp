#include "datatypes/trajectory.h"
#include "mapping/local_mapper.h"
#include "test.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace Kompass;

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

void printMatrix(const Eigen::MatrixXf &matrix) {
  std::cout << "Matrix (" << matrix.rows() << "x" << matrix.cols() << "):\n";
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      std::cout << matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

void printMatrix(const Eigen::MatrixXi &matrix) {
  std::cout << "Matrix (" << matrix.rows() << "x" << matrix.cols() << "):\n";
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      std::cout << matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
}

Eigen::VectorXd stdVectorToEigenVector(const std::vector<double> &std_vector) {
  Eigen::Map<const Eigen::VectorXd> eigen_vector(std_vector.data(),
                                                 std_vector.size());
  return eigen_vector;
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
    std::cerr << "Invalid shape specified. Use 'circle', 'right_corner', or "
                 "'random_points'.\n";
  }

  return {ranges, angles};
}

// Example usage
int main() {

  double angle_increment = 0.1; // Example increment in radians
  double corner_distance = 0.5; // Example max range for the right corner
  int random_points = 50;       // Example number of random points

  int grid_width = 10;
  int grid_height = 10;
  float grid_res = 0.1;
  float actual_size = grid_width * grid_res;

  // Init the grid
  Eigen::MatrixXi gridData(grid_width, grid_height);
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));

  Eigen::MatrixXf gridDataProb(grid_width, grid_height);
  Eigen::MatrixXf prevGridDataProb(grid_width, grid_height);

  // Central point
  int x = std::round(grid_width / 2) - 1;
  int y = std::round(grid_height / 2) - 1;
  Eigen::Vector2i centralPoint(x, y);
  LOG_INFO("Central point:", x, ",", y);

  // scan model
  float pPrior = 0.6, pOccupied = 0.9;
  float pEmpty = 1 - pOccupied;
  float rangeSure = 0.1, rangeMax = 20.0, wallSize = 0.2;
  int maxNumThreads = 1;
  gridDataProb.fill(pPrior);
  prevGridDataProb.fill(pPrior);

  // limit ranges to an elipse inscribing the grid
  double limit = grid_width >= grid_height ? grid_width * grid_res * sqrt(2)
                                           : grid_height * grid_res * sqrt(2);

  LOG_INFO("Limit Circle Radius:", limit);

  int maxPointsPerLine =
      (limit / grid_res) * 1.5; // on average 1.5 cells highlighted per step
  LOG_INFO("Max Steps in Grid inscribing the limit circle:", maxPointsPerLine);

  // Vector to store the filtered ranges
  std::vector<double> filtered_ranges;

  // Generate circle scan with radius 1.0
  double radius = 0.3; // Example radius for the circle
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
    Mapping::scanToGrid(circle_scan.angles, filtered_ranges, gridData,
                        gridDataProb, centralPoint, grid_res, {0.0, 0.0, 0.0},
                        0.0, prevGridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                        rangeMax, wallSize, maxPointsPerLine, maxNumThreads);
  }

  int occ_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  int free_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  int unknown_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  printMatrix(gridData);
  printMatrix(gridDataProb);

  // Generate circle scan with radius 0.5
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
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
    Mapping::scanToGrid(circle_scan.angles, filtered_ranges, gridData,
                        gridDataProb, centralPoint, grid_res, {0.0, 0.0, 0.0},
                        0.0, prevGridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                        rangeMax, wallSize, maxPointsPerLine, maxNumThreads);
  }
  occ_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  free_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  unknown_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  printMatrix(gridData);
  printMatrix(gridDataProb);

  // Generate circle scan with radius 10.5
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  radius = 10.5; // Example radius for the circle
  circle_scan = generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           "and grid of width: ", actual_size);
  filtered_ranges.resize(circle_scan.ranges.size());
  for (size_t i = 0; i < circle_scan.ranges.size(); ++i) {
    filtered_ranges[i] = std::min(limit, circle_scan.ranges[i]);
  }
  {
    Timer timer;
    Mapping::scanToGrid(circle_scan.angles, filtered_ranges, gridData,
                        gridDataProb, centralPoint, grid_res, {0.0, 0.0, 0.0},
                        0.0, prevGridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                        rangeMax, wallSize, maxPointsPerLine, maxNumThreads);
  }
  occ_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  free_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::EMPTY));
  unknown_points = countPointsInGrid(
      gridData, static_cast<int>(Mapping::OccupancyType::UNEXPLORED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  printMatrix(gridData);
  printMatrix(gridDataProb);

  return 0;
}
