#include "utils/collision_check.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/src/Geometry/Transform.h>
#include <LinearMath/btQuaternion.h>
#include <LinearMath/btVector3.h>
#include <memory>
#include <vector>

namespace Kompass {

// Constructor
CollisionChecker::CollisionChecker(
    const ShapeType robot_shape_type,
    const std::vector<float> &robot_dimensions,
    const std::array<float, 3> &sensor_position_body,
    const std::array<float, 4> &sensor_rotation_body,
    const double octree_resolution) {

  // Obstacles resolution
  octree_resolution_ = octree_resolution;
  // Initialize Bullet collision configuration
  m_collisionConfiguration =
      std::make_unique<btDefaultCollisionConfiguration>();
  m_dispatcher =
      std::make_unique<btCollisionDispatcher>(m_collisionConfiguration.get());
  m_broadphase = std::make_unique<btDbvtBroadphase>();
  m_solver = std::make_unique<btSequentialImpulseConstraintSolver>();
  m_collisionWorld = std::make_unique<btDiscreteDynamicsWorld>(
      m_dispatcher.get(), m_broadphase.get(), m_solver.get(),
      m_collisionConfiguration.get());

  // Create the robot object
  body_.dimensions = robot_dimensions;

  switch (robot_shape_type) {
  case ShapeType::BOX:
    body_.robotShape = std::make_unique<btBoxShape>(btVector3{
        robot_dimensions[0], robot_dimensions[1], robot_dimensions[2]});
    robotHeight_ = body_.dimensions.at(2);
    robotRadius_ = std::sqrt(pow(body_.dimensions.at(0), 2) +
                             pow(body_.dimensions.at(1), 2)) /
                   2;
    break;
  case ShapeType::SPHERE:
    body_.robotShape =
        std::make_shared<btSphereShape>(robot_dimensions[0]); // radius
    robotHeight_ = body_.dimensions.at(0) * 2.0;
    robotRadius_ = body_.dimensions.at(0);
    break;
  case ShapeType::CYLINDER:
    // Cylinder requires specifying axis and dimensions
    // dimensions.x() = radius, dimensions.y() = height, dimensions.z() is not
    // used
    body_.robotShape = std::make_shared<btCylinderShape>(
        btVector3{robot_dimensions[0], robot_dimensions[1], 0.0});
    robotHeight_ = body_.dimensions.at(1);
    robotRadius_ = body_.dimensions.at(0);
    break;
  default:
    throw std::runtime_error("Invalid robot shape type");
  }

  // Init the sensor position w.r.t body
  sensor_tf_body_ =
      getTransformation(Eigen::Quaternionf(sensor_rotation_body.data()),
                        Eigen::Vector3f(sensor_position_body.data()));

  sensor_tf_world_ = sensor_tf_body_;

  // Create a temporary collision object for the robot
  m_robotObject = new btCollisionObject();
  m_robotObject->setCollisionShape(body_.robotShape.get());
}

// Destructor to clean up resources
CollisionChecker::~CollisionChecker() {
  delete m_robotObject;
  clearObstacles();
}

void CollisionChecker::resetOctreeResolution(const double resolution) {
  octree_resolution_ = resolution;
}

void CollisionChecker::updateState(const Path::State current_state) {

  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation = eulerToRotationMatrix(0.0, 0.0, current_state.yaw);
  body_.tf = getTransformation(
      rotation, Eigen::Vector3f(current_state.x, current_state.y, 0.0));
  btTransform robotTransform;
  Eigen::Quaternionf quat(rotation);
  robotTransform.setRotation(
      btQuaternion(quat.x(), quat.y(), quat.z(), quat.w()));
  robotTransform.setOrigin(btVector3{static_cast<float>(current_state.x),
                                     static_cast<float>(current_state.y), 0.0});

  m_robotObject->setWorldTransform(robotTransform);
}

void CollisionChecker::updateState(const double x, const double y,
                                   const double yaw) {

  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation = eulerToRotationMatrix(0.0, 0.0, yaw);
  body_.tf = getTransformation(rotation, Eigen::Vector3f(x, y, 0.0));
  btTransform robotTransform;
  Eigen::Quaternionf quat(rotation);
  robotTransform.setRotation(
      btQuaternion(quat.x(), quat.y(), quat.z(), quat.w()));
  robotTransform.setOrigin(
      btVector3{static_cast<float>(x), static_cast<float>(y), 0.0});

  m_robotObject->setWorldTransform(robotTransform);
}

void CollisionChecker::updateScan(
    const std::vector<double> &ranges,
    const std::vector<double> &angles) { // Clear previous obstacles

  // Size checks
  if (ranges.size() != angles.size()) {
    throw std::invalid_argument("Ranges and angles must have the same length");
  }

  clearObstacles();

  // Transform the sensor position to the world frame
  sensor_tf_world_ = sensor_tf_body_ * body_.tf;

  // Transform height to sensor frame
  float height_in_sensor = -sensor_tf_body_.translation()[2] / 2;

  btTransform obstacleTransform;
  float x, y;
  // Process each laser scan point
  for (size_t i = 0; i < ranges.size(); ++i) {

    // Convert polar (range, angle) to Cartesian (x, y)
    x = ranges[i] * std::cos(angles[i]);
    y = ranges[i] * std::sin(angles[i]);

    auto point_in_world =
        sensor_tf_world_ * Eigen::Vector3f{x, y, height_in_sensor};

    // Create a small sphere to represent the obstacle point
    auto obstacleShape =
        std::make_unique<btSphereShape>(octree_resolution_ / 2);

    // Create collision object for the obstacle
    btCollisionObject *obstacleObject = new btCollisionObject();
    obstacleObject->setCollisionShape(obstacleShape.get());

    // Set obstacle position
    obstacleTransform.setIdentity();
    obstacleTransform.setOrigin(
        btVector3(point_in_world.x(), point_in_world.y(), point_in_world.z()));
    obstacleObject->setWorldTransform(obstacleTransform);

    // Add to collision world
    m_collisionWorld->addCollisionObject(obstacleObject);
    m_obstacles.push_back(std::move(obstacleShape));
    m_obstacleObjects.push_back(obstacleObject);
  }
}

void CollisionChecker::updatePointCloud(const std::vector<Path::Point> &cloud,
                                        bool inRobotFrame) {
  if (inRobotFrame) {
    // If data is provided in the robot frame (not world frame) -> set the
    // transform
    sensor_tf_world_ = sensor_tf_body_ * body_.tf;
  } else {
    sensor_tf_world_ = Eigen::Isometry3f::Identity();
  }
  clearObstacles();

  btTransform obstacleTransform;

  for (const auto &point : cloud) {

    auto point_in_world =
        sensor_tf_world_ * Eigen::Vector3f{point.x(), point.y(), point.z()};

    // Create a small sphere to represent the obstacle point
    auto obstacleShape =
        std::make_unique<btSphereShape>(octree_resolution_ / 2);

    // Create collision object for the obstacle
    btCollisionObject *obstacleObject = new btCollisionObject();
    obstacleObject->setCollisionShape(obstacleShape.get());

    // Set obstacle position
    obstacleTransform.setIdentity();
    obstacleTransform.setOrigin(
        btVector3(point_in_world.x(), point_in_world.y(), point_in_world.z()));
    obstacleObject->setWorldTransform(obstacleTransform);

    // Add to collision world
    m_collisionWorld->addCollisionObject(obstacleObject);
    m_obstacles.push_back(std::move(obstacleShape));
    m_obstacleObjects.push_back(obstacleObject);
  }
}

bool CollisionChecker::checkCollisions() {

  MyContactResult callback = MyContactResult();
  m_collisionWorld->addCollisionObject(m_robotObject);
  m_collisionWorld->contactTest(m_robotObject, callback);

  m_collisionWorld->removeCollisionObject(m_robotObject);
  return callback.hasCollided;
}

bool CollisionChecker::checkCollisions(const Path::State current_state,
                                       const bool multi_threading) {
  if (multi_threading) {
    // Lock the mutex to ensure thread-safe access to m_obstacleObjects
    std::lock_guard<std::mutex> lock(m_obstacleMutex);
    auto collisionConfiguration =
        std::make_unique<btDefaultCollisionConfiguration>();
    auto dispatcher =
        std::make_unique<btCollisionDispatcher>(collisionConfiguration.get());
    auto broadphase = std::make_unique<btDbvtBroadphase>();
    auto solver = std::make_unique<btSequentialImpulseConstraintSolver>();
    auto collisionWorld = std::make_unique<btDiscreteDynamicsWorld>(
        dispatcher.get(), broadphase.get(), solver.get(),
        collisionConfiguration.get());

    auto robotObject = std::make_unique<btCollisionObject>();
    robotObject->setCollisionShape(body_.robotShape.get());
    Eigen::Matrix3f rotation =
        eulerToRotationMatrix(0.0, 0.0, current_state.yaw);
    btTransform robotTransform;
    Eigen::Quaternionf quat(rotation);
    robotTransform.setRotation(
        btQuaternion(quat.x(), quat.y(), quat.z(), quat.w()));
    robotTransform.setOrigin(btVector3{static_cast<float>(current_state.x),
                                       static_cast<float>(current_state.y),
                                       0.0});

    robotObject->setWorldTransform(robotTransform);

    std::vector<std::unique_ptr<btCollisionObject>> obsObjects;
    for (auto *obstacle : m_obstacleObjects) {
      obsObjects.emplace_back(new btCollisionObject(*obstacle));
    }

    MyContactResult callback = MyContactResult();
    collisionWorld->addCollisionObject(robotObject.get());

    // Add obstacles to the collision world and check for collisions
    for (auto &obstacleObject : obsObjects) {
      collisionWorld->addCollisionObject(obstacleObject.get());
    }

    collisionWorld->contactTest(robotObject.get(), callback);

    // Remove obstacles from the collision world and delete them
    for (auto &obstacleObject : obsObjects) {
      collisionWorld->removeCollisionObject(obstacleObject.get());
    }
    collisionWorld->removeCollisionObject(robotObject.get());

    obsObjects.clear();

    return callback.hasCollided;

  } else {
    updateState(current_state);
    return checkCollisions();
  }
}

bool CollisionChecker::checkCollisions(const std::vector<double> &ranges,
                                       const std::vector<double> &angles) {
  updateScan(ranges, angles);
  return checkCollisions(); /*  */
}

// Clear all obstacles
void CollisionChecker::clearObstacles() {
  for (auto *obstacleObject : m_obstacleObjects) {
    m_collisionWorld->removeCollisionObject(obstacleObject);
    delete obstacleObject;
  }
  m_obstacleObjects.clear();
  m_obstacles.clear();
}

} // namespace Kompass
