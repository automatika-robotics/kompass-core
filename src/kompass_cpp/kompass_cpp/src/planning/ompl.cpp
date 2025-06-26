#include "planning/ompl.h"
#include "utils/logger.h"

namespace Kompass {
namespace Planning {
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
}

OMPL2DGeometricPlanner::~OMPL2DGeometricPlanner() {}

void OMPL2DGeometricPlanner::setupProblem(
    double start_x, double start_y, double start_yaw, double goal_x,
    double goal_y, double goal_yaw,
    const std::vector<Eigen::Vector3f> &map_3d) {
  setup_->clear();
  collision_checker_->update3DMap(map_3d);
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

  setup_->setStartAndGoalStates(start, goal);
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
