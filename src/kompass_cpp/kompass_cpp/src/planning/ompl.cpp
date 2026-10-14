#include "planning/ompl.h"
#include "utils/logger.h"
#include <cmath>
#include <ompl/base/goals/GoalState.h>

namespace Kompass {
namespace Planning {
namespace {
/**
 * @brief A goal reached by standing on it, whatever way the robot faces.
 *
 * The threshold OMPL checks a state against is a distance in the whole state
 * space, and SE(2) weights yaw at 0.5: with a threshold of one map cell the
 * heading would have to match within a tenth of a radian, which a sampling
 * planner rarely hits, so every solve would come back approximate. A 2D
 * navigation goal is a place, not a pose: arrival is judged on position, and
 * the heading at the end belongs to whoever drives the path.
 */
class PositionGoal : public ompl::base::GoalState {
public:
  using ompl::base::GoalState::GoalState;

  double distanceGoal(const ompl::base::State *state) const override {
    const auto *reached = state->as<ompl::base::SE2StateSpace::StateType>();
    const auto *wanted = getState()->as<ompl::base::SE2StateSpace::StateType>();
    return std::hypot(reached->getX() - wanted->getX(),
                      reached->getY() - wanted->getY());
  }
};
} // namespace
OMPL2DGeometricPlanner::OMPL2DGeometricPlanner(
    const CollisionChecker::ShapeType &robot_shape_type,
    const std::vector<float> &robot_dimensions,
    const ompl::geometric::SimpleSetup &setup, const float map_resolution) {
  setup_ = ompl::geometric::SimpleSetupPtr(
      std::make_shared<ompl::geometric::SimpleSetup>(setup));
  setup_->setStateValidityChecker([this](const ompl::base::State *state) {
    return this->state_validity_checker(state);
  });
  collision_checker_ = std::make_shared<CollisionChecker>(
      robot_shape_type, robot_dimensions, Eigen::Vector3f(0.0, 0.0, 0.0),
      Eigen::Quaternionf(1.0, 0.0, 0.0, 0.0), map_resolution);
  goal_threshold_ = map_resolution;
}

OMPL2DGeometricPlanner::~OMPL2DGeometricPlanner() {}

void OMPL2DGeometricPlanner::setMap(Span<Path::Point> map_3d) {
  // The planning map is already in the world frame, and this checker is built
  // with an identity sensor mount, so an identity capture pose makes the
  // sensor frame coincide with the world frame. Passing it explicitly also
  // keeps the map off whatever pose the previous solve's validity checks left
  // behind in the checker.
  collision_checker_->updateSensorData(map_3d, Path::State());
}

void OMPL2DGeometricPlanner::setupProblem(double start_x, double start_y,
                                          double start_yaw, double goal_x,
                                          double goal_y, double goal_yaw,
                                          Span<Path::Point> map_3d) {
  setMap(map_3d);
  setupProblem(start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw);
}

void OMPL2DGeometricPlanner::setupProblem(double start_x, double start_y,
                                          double start_yaw, double goal_x,
                                          double goal_y, double goal_yaw) {
  setup_->clear();
  ompl::base::ScopedState<ompl::base::SE2StateSpace> start(
      setup_->getStateSpace());
  ompl::base::ScopedState<ompl::base::SE2StateSpace> goal(
      setup_->getStateSpace());
  start->setX(start_x);
  start->setY(start_y);
  start->setYaw(start_yaw);

  goal->setX(goal_x);
  goal->setY(goal_y);
  goal->setYaw(goal_yaw);

  setup_->setStartState(start);
  auto goal_region =
      std::make_shared<PositionGoal>(setup_->getSpaceInformation());
  goal_region->setState(goal);
  goal_region->setThreshold(goal_threshold_);
  setup_->setGoal(goal_region);
}

void OMPL2DGeometricPlanner::setSpaceBoundsFromMap(const float origin_x,
                                                   const float origin_y,
                                                   const int width,
                                                   const int height,
                                                   const float resolution) {
  auto bounds = ompl::base::RealVectorBounds(2);
  bounds.setLow(0, origin_x);
  bounds.setLow(1, origin_y);
  bounds.setHigh(0, origin_x + float(resolution * width));
  bounds.setHigh(1, origin_y + float(resolution * height));
  setup_->getStateSpace()->as<ompl::base::SE2StateSpace>()->setBounds(bounds);
  // A plan is no more precise than one cell of the map it is planned on
  goal_threshold_ = resolution;
}

bool OMPL2DGeometricPlanner::solve(double planning_timeout) {
  auto state = setup_->solve(planning_timeout);
  if (state == ompl::base::PlannerStatus::APPROXIMATE_SOLUTION or
      state == ompl::base::PlannerStatus::EXACT_SOLUTION) {
    gotSolution_ = true;
    setup_->simplifySolution();
  } else {
    LOG_ERROR("OMPL planner failed to find path, status = ", state.asString());
    gotSolution_ = false;
  }
  return gotSolution_;
}

std::optional<ompl::geometric::PathGeometric>
OMPL2DGeometricPlanner::getPath() const {
  if (gotSolution_) {
    return setup_->getSolutionPath();
  }
  return std::nullopt;
}

float OMPL2DGeometricPlanner::getCost() const {
  auto solution = this->getPath();
  if (solution) {
    auto optimization_objective = setup_->getOptimizationObjective();
    ompl::base::Cost cost = solution->cost(optimization_objective);
    return float(cost.value());
  }
  return 0.0f;
}

bool OMPL2DGeometricPlanner::state_validity_checker(
    const ompl::base::State *state) {
  bool is_state_valid = setup_->getSpaceInformation()->satisfiesBounds(state);
  auto se2_state = state->as<ompl::base::SE2StateSpace::StateType>();
  double x = se2_state->getX();
  double y = se2_state->getY();
  double yaw = se2_state->getYaw();

  collision_checker_->updateState(x, y, yaw);

  bool has_collisions = collision_checker_->checkCollisions();

  return is_state_valid and not has_collisions;
}
} // namespace Planning

} // namespace Kompass
