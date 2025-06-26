#pragma once

#include "utils/collision_check.h"
#include <fcl/fcl.h>
#include <ompl/base/Cost.h>
#include <ompl/base/Planner.h>
#include <ompl/base/PlannerStatus.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/SimpleSetup.h>
#include <optional>

namespace Kompass {

namespace Planning {

class OMPL2DGeometricPlanner {
public:
  OMPL2DGeometricPlanner(const CollisionChecker::ShapeType &robot_shape_type,
                         const std::vector<float> &robot_dimensions,
                         const ompl::geometric::SimpleSetup &setup,
                         const float map_resolution = 0.01);
  ~OMPL2DGeometricPlanner();

  /**
   * @brief Setup OMPL planning problem
   *
   * @param start_x
   * @param start_y
   * @param start_yaw
   * @param goal_x
   * @param goal_y
   * @param goal_yaw
   * @param map_3d
   */
  void setupProblem(double start_x, double start_y, double start_yaw,
                    double goal_x, double goal_y, double goal_yaw,
                    const std::vector<Eigen::Vector3f> &map_3d);
  /**
   * @brief Solve the planning problem
   *
   * @param planning_timeout
   * @return true
   * @return false
   */
  bool solve(double planning_timeout);

  /**
   * @brief Get the Path object
   *
   * @return std::optional<ompl::geometric::PathGeometric>
   */
  std::optional<ompl::geometric::PathGeometric> getPath() const;

  /**
   * @brief Get the Cost object
   *
   * @return float
   */
  float getCost() const;

  /**
   * @brief Set the Space Bounds From Map object
   *
   * @param origin_x
   * @param origin_y
   * @param width
   * @param height
   * @param resolution
   */
  void setSpaceBoundsFromMap(const float origin_x, const float origin_y,
                             const int width, const int height,
                             const float resolution);

private:
  bool gotSolution_ = false;
  ompl::geometric::SimpleSetupPtr setup_;
  std::shared_ptr<CollisionChecker> collision_checker_;

  /**
   * @brief State validity checking function (uses collision checker)
   *
   * @param state
   * @return true
   * @return false
   */
  bool state_validity_checker(const ompl::base::State *state);
};

} // namespace Planning

} // namespace Kompass
