#pragma once

#include "control.h"
#include "datatypes/path.h"
#include <cmath>
#include <vector>

namespace Kompass {

namespace Control {
/**
 * @brief Trajectory infomration: Path + Velocity
 *
 */
struct Trajectory {
  std::vector<Velocity> velocity;
  Path::Path path;
};

/**
 * @brief Trajectory control result (Local planners result), contains a boolean
 * indicating if the trajectory is found, the resulting trajectory and its const
 *
 */
struct TrajSearchResult {
  bool isTrajFound;
  double trajCost;
  Trajectory trajectory;
};

/**
 * @brief Struct for LaserScan data
 *
 */
struct LaserScan {
  std::vector<double> ranges;
  std::vector<double> angles;

  LaserScan(std::vector<double> ranges, std::vector<double> angles)
      : ranges(ranges), angles(angles) {}
};

struct Point3D {
  double x;
  double y;
  double z;

  Point3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
};

}; // namespace Control
} // namespace Kompass
