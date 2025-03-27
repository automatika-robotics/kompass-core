#pragma once

#include "BulletCollision/CollisionDispatch/btManifoldResult.h"
#include "datatypes/path.h"
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <bullet/btBulletCollisionCommon.h>
#include <memory>
#include <vector>

namespace Kompass {

class CollisionChecker {
public:
  enum class ShapeType { SPHERE, CYLINDER, BOX };

  struct Body {
    std::shared_ptr<btCollisionShape> robotShape;
    std::vector<float>
        dimensions; // For cylinder: dimensions[0]=radius, dimensions[1]=height.
                    // For box: dimensions[0]=x, dimensions[1]=y,
                    // dimensions[2]=z.
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    ; // Transformation matrix of the body
  };

  class MyContactResult : public btCollisionWorld::ContactResultCallback {
  public:
    bool hasCollided = false;

    MyContactResult() : btCollisionWorld::ContactResultCallback() {}

    btScalar addSingleResult(btManifoldPoint &cp,
                             const btCollisionObjectWrapper *colObj0Wrap,
                             int partId0, int index0,
                             const btCollisionObjectWrapper *colObj1Wrap,
                             int partId1, int index1) {
      hasCollided = true;
      return 1.0; // Continue checking
    }
  };

  // Constructor
  CollisionChecker(const ShapeType robot_shape_type,
                   const std::vector<float> &robot_dimensions,
                   const std::array<float, 3> &sensor_position_body,
                   const std::array<float, 4> &sensor_rotation_body,
                   const double octree_resolution = 0.01);

  // Destructor to clean up resources
  ~CollisionChecker();

  /**
   * @brief Reset the resolution of the obstacles Octree
   *
   * @param resolution
   */
  void resetOctreeResolution(const double resolution);

  /**
   * @brief Update the current state of the robot
   *
   * @param current_state
   */
  void updateState(const Path::State current_state);

  /**
   * @brief Update the current state of the robot with 2D pose (x, y, yaw)
   *
   * @param x
   * @param y
   * @param yaw
   */
  void updateState(const double x, const double y, const double yaw);

  /**
   * @brief Update the sensor input from laser scan data
   *
   * @param ranges
   * @param angles
   */
  void updateScan(const std::vector<double> &ranges,
                  const std::vector<double> &angles);

  /**
   * @brief Update the sensor input from laser scan data
   *
   * @param ranges
   * @param angles
   */
  void updateScan(const std::vector<double> &ranges, double angle_min,
                  double angle_increment);

  // void updateLocalMap(const std::vector<std::pair<float, float>>
  // &occupancyMap,
  //                     float cellSize);

  /**
   * @brief Update the sensor input from PointCloud like struct
   * std::vector<Control::Point3D>
   *
   * @param cloud
   */
  void updatePointCloud(const std::vector<Path::Point> &cloud,
                        bool inRobotFrame = false);

  /**
   * @brief Get the Min Distance from the Octree (sensor data Octomap)
   *
   * @return float
   */
  // double getMinDistance();

  /**
   * @brief Get the Min Distance object from given LaserScan data
   *
   * @param ranges
   * @param angle_min
   * @param angle_increment
   * @param height
   * @return float
   */
  // double getMinDistance(const std::vector<double> &ranges, double angle_min,
  //                       double angle_increment);

  // Check collision between robot and obstacles
  bool checkCollisions();

  bool checkCollisions(const Path::State current_state,
                       const bool multi_threading = false);

  bool checkCollisions(const std::vector<double> &ranges,
                       const std::vector<double> &angles);

protected:
  double robotHeight_{1.0}, robotRadius_, octree_resolution_;
  // sensor tf with respect to the world
  Eigen::Isometry3f sensor_tf_world_ = Eigen::Isometry3f::Identity();

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot
private:
  // Robot body geometry object
  Body body_;
  // Bullet collision world components
  std::shared_ptr<btDefaultCollisionConfiguration> m_collisionConfiguration;
  std::shared_ptr<btCollisionDispatcher> m_dispatcher;
  std::shared_ptr<btDbvtBroadphase> m_broadphase;
  std::shared_ptr<btConstraintSolver> m_solver;
  std::shared_ptr<btDiscreteDynamicsWorld> m_collisionWorld;
  btCollisionObject *m_robotObject;

  // Stored obstacles
  std::vector<std::shared_ptr<btCollisionShape>> m_obstacles;
  std::vector<btCollisionObject *> m_obstacleObjects;

  void clearObstacles();
};
} // namespace Kompass
