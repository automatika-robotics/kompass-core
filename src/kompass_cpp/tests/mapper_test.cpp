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

  int grid_width = 20;
  int grid_height = 20;
  float grid_res = 0.1;
  float actual_size = grid_width * grid_res;

  // Init the grid
  Eigen::MatrixXi gridData(grid_height, grid_width);
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNKNOWN));

  Eigen::MatrixXf gridDataProb(grid_height, grid_width);
  gridDataProb.fill(static_cast<int>(Mapping::OccupancyType::UNKNOWN));

  // Central point
  int x = static_cast<int>(std::round(grid_width / 2.0)) - 1;
  int y = static_cast<int>(std::round(grid_height / 2.0)) - 1;
  Eigen::Vector2i centralPoint(x, y);
  std::cout << "Central point" << x << ", " << y << std::endl;

  // scan model
  float pPrior = 0.6, pOccupied = 0.9;
  float pEmpty = 1 - pOccupied;
  float rangeSure = 0.1, rangeMax = 20.0, wallSize = 0.075;
  float oddLogPPrior = static_cast<float>(std::log(pPrior / (1.0f - pPrior)));
  int maxNumThreads = 10;

  Timer timer;

  // Generate circle scan with radius 1.0
  double radius = 1.0; // Example radius for the circle
  Control::LaserScan circle_scan =
      generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           ", and grid of width: ", actual_size);
  Mapping::scanToGrid(circle_scan.angles, circle_scan.ranges, gridData,
                      gridDataProb, centralPoint, grid_res, {0.0, 0.0, 0.0},
                      0.0, gridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                      rangeMax, wallSize, oddLogPPrior, maxNumThreads);

  int occ_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  int free_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::FREE));
  int unknown_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  printMatrix(gridData);

  // Generate circle scan with radius 0.5
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNKNOWN));
  radius = 0.5; // Example radius for the circle
  circle_scan = generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           ", and grid of width: ", actual_size);
  Mapping::scanToGrid(circle_scan.angles, circle_scan.ranges, gridData,
                      gridDataProb, centralPoint, grid_res, {0.0, 0.0, 0.0},
                      0.0, gridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                      rangeMax, wallSize, oddLogPPrior, maxNumThreads);
  occ_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  free_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::FREE));
  unknown_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  printMatrix(gridData);

  // Generate circle scan with radius 3.5
  gridData.fill(static_cast<int>(Mapping::OccupancyType::UNKNOWN));
  radius = 10.5; // Example radius for the circle
  circle_scan = generateLaserScan(angle_increment, "circle", radius);
  LOG_INFO("Testing with circle points at distance: ", radius,
           ", and grid of width: ", actual_size);
  Mapping::scanToGrid(circle_scan.angles, circle_scan.ranges, gridData,
                      gridDataProb, centralPoint, grid_res, {0.0, 0.0, 0.0},
                      0.0, gridDataProb, pPrior, pEmpty, pOccupied, rangeSure,
                      rangeMax, wallSize, oddLogPPrior, maxNumThreads);
  occ_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  free_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::FREE));
  unknown_points = countPointsInGrid(gridData, static_cast<int>(Mapping::OccupancyType::OCCUPIED));
  LOG_INFO("Number of occupied cells: ", occ_points);
  LOG_INFO("Number of free cells: ", free_points);
  LOG_INFO("Number of unknown cells: ", unknown_points);
  printMatrix(gridData);

  return 0;
}
