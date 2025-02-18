#pragma once

#include "control.h"
#include "datatypes/path.h"
#include <cmath>
#include <vector>

namespace Kompass {

namespace Control {
/**
 * @brief Trajectory information: Path + Velocity
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

} // namespace Control
} // namespace Kompass
