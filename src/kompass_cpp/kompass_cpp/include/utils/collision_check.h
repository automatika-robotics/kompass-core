#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/geometry/octree/octree.h>
#include <fcl/narrowphase/collision_object.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>

#include "datatypes/path.h"

namespace Kompass {
/**
 * @brief Collision Checker using FCL (Flexible Collisions Library)
 * The collision checker supports sensor input data from PointCloud and
 * LaserScan
 *
 */
class CollisionChecker {
public:
  enum class ShapeType { CYLINDER, BOX, SPHERE };

  struct Body {
    ShapeType shapeType = ShapeType::BOX;
    std::vector<float>
        dimensions; // For cylinder: dimensions[0]=radius,
                    // dimensions[1]=height. For box: dimensions[0]=x,
                    // dimensions[1]=y, dimensions[2]=z.
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    ; // Transformation matrix of the body
  };

  /**
   * @brief Construct a new Collision Checker object
   *
   * @param robotShapeType    Type of the robot shape geometry
   * @param robotDimensions   Corresponding geometry dimensions
   * @param sensorPositionWRTbody         Position of the sensor w.r.t the
   * robot body - Considered constant
   * @param octreeRes         Resolution of the constructed OctTree
   */
  CollisionChecker(const ShapeType robot_shape_type,
                   const std::vector<float> &robot_dimensions,
                   const Eigen::Vector3f &sensor_position_body,
                   const Eigen::Quaternionf &sensor_rotation_body,
                   const double octree_resolution = 0.01);

  /**
   * @brief Destroy the Collision Checker object
   *
   */
  ~CollisionChecker() = default;

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
   * @brief Update the sensor input from Map points (Eigen::Matrix3Xf) data
   *
   * @param points
   * @param global_frame
   */
  void update3DMap(const std::vector<Eigen::Vector3f> &points,
                   const bool global_frame = true);

  /**
   * @brief Update the sensor input from PointCloud like struct
   * std::vector<Control::Point3D>
   *
   * @param cloud
   */
  void updatePointCloud(const std::vector<Path::Point> &cloud,
                        const bool global_frame = true);

  /**
   * @brief Check collisions between the robot and the constructed OctTree
   *
   * @return true
   * @return false
   */
  bool checkCollisionsOctree();

  /**
   * @brief Get the Min Distance from the Octree (sensor data Octomap)
   *
   * @return float
   */
  float getMinDistance();

  /**
   * @brief Get the Min Distance object from given LaserScan data
   *
   * @param ranges
   * @param angle_min
   * @param angle_increment
   * @param height
   * @return float
   */
  float getMinDistance(const std::vector<double> &ranges, double angle_min,
                       double angle_increment, double height = 0.1);

  /**
   * @brief Get the Min Distance object from given LaserScan data
   *
   * @param ranges
   * @param angles
   * @param height
   * @return float
   */
  float getMinDistance(const std::vector<double> &ranges,
                       const std::vector<double> &angles, double height = 0.1);

  /**
   * @brief Check collisions between the robot and previously set sensor data
   *
   * @return true
   * @return false
   */
  bool checkCollisions();

  /**
   * @brief Check collisions between the robot and given PointCloud data
   *
   * @param cloud
   * @return true
   * @return false
   */
  bool checkCollisions(const std::vector<Path::Point> &cloud);

  /**
   * @brief Check collisions between the robot and given LaserScan data
   * defined by a vector of ranges values, minimum scan angle and scan angle
   * step
   *
   * @param ranges
   * @param angle_min
   * @param angle_increment
   * @param height        Height of the constructed OctTree map, defaults to
   * 0.1
   * @return true
   * @return false
   */
  bool checkCollisions(const std::vector<double> &ranges, double angle_min,
                       double angle_increment, double height = 0.1);

  /**
   * @brief Check collisions between the robot and given LaserScan data
   * defined by a vector of ranges values, and a vector of angles values
   *
   * @param ranges
   * @param angles
   * @param height        Height of the constructed OctTree map, defaults to
   * 0.1
   * @return true
   * @return false
   */
  bool checkCollisions(const std::vector<double> &ranges,
                       const std::vector<double> &angles, double height = 0.1);

  /**
   * @brief Check collisions between the given robot state and existing
   * Octomap data defined
   *
   * @param current_state
   * @return true
   * @return false
   */
  bool checkCollisions(const Path::State current_state);

  float getRadius() const;

protected:
  double robotHeight_{1.0}, robotRadius_;
  // sensor tf with respect to the world
  Eigen::Isometry3f sensor_tf_world_ = Eigen::Isometry3f::Identity();

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot

private:
  // Collision Manager
  std::unique_ptr<fcl::DynamicAABBTreeCollisionManagerf> collManager_;

  // Robot body geometry object
  std::shared_ptr<Body> body;
  std::shared_ptr<fcl::CollisionGeometryf> bodyGeometry_;

  // Robot body collision object pointer
  std::shared_ptr<fcl::CollisionObjectf> bodyObjPtr_ = nullptr;

  // Octree collision object pointers
  std::shared_ptr<octomap::OcTree>
      octTree_; // Octomap octree used to get data from laserscan
                // or pointcloud and convert it to an Octree
  std::shared_ptr<fcl::OcTreef> fclTree_ =
      nullptr; // FCL Octree updated after converting the Octomap octree
  // (required for creating the collision object)
  octomap::Pointcloud octomapCloud_;
  std::unique_ptr<fcl::CollisionObjectf>
      OctreeCollObj_; // Octree collision object

  double octree_resolution_{0.01};

  /**
   * @brief Updates the Octree collision object pointer
   *
   */
  void updateOctreePtr();

  /**
   * @brief Helper method to generate a vector of fcl::Box collision objects
   * from an fcl::Octree
   *
   * @param boxes
   * @param tree
   */
  std::vector<fcl::CollisionObjectf *>
  generateBoxesFromOctomap(fcl::OcTreef &tree);

  /**
   * @brief Helper method to convert PointCloud data to an Octomap
   *
   * @param cloud
   */
  void convertPointCloudToOctomap(const std::vector<Path::Point> &cloud,
                                  const bool global_frame = true);

  /**
   * @brief Helper method to convert LaserScan data to an OctoMap. LaserScan
   * data defined by a vector of ranges values, minimum scan angle and scan
   * angle step
   *
   * @param ranges
   * @param angle_min
   * @param angle_increment
   * @param height
   */
  void convertLaserScanToOctomap(const std::vector<double> &ranges,
                                 double angle_min, double angle_increment,
                                 double height = 0.1);

  /**
   * @brief Helper method to convert LaserScan data to an OctoMap. LaserScan
   * data defined by a vector of ranges values, and a vector of angles values
   *
   * @param ranges
   * @param angles
   * @param height
   */
  void convertLaserScanToOctomap(const std::vector<double> &ranges,
                                 const std::vector<double> &angles,
                                 double height = 0.1);
};
} // namespace Kompass
