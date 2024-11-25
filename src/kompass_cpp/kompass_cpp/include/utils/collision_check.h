#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>
#include <memory>
#include <vector>

#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/fcl.h>
#include <fcl/geometry/octree/octree.h>
#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "datatypes/path.h"
#include "datatypes/trajectory.h"

namespace Kompass {
/**
 * @brief Collision Checker using FCL (Flexible Collisions Library)
 * The collision checker supports sensor input data from PoinCloud and LaserScan
 *
 */
class CollisionChecker {
public:
  enum class ShapeType { CYLINDER, BOX };

  struct Body {
    ShapeType shapeType = ShapeType::BOX;
    std::vector<float>
        dimensions; // For cylinder: dimensions[0]=radius, dimensions[1]=height.
                    // For box: dimensions[0]=x, dimensions[1]=y,
                    // dimensions[2]=z.
    Eigen::Isometry3f tf = Eigen::Isometry3f::Identity();
    ; // Transformation matrix of the body
  };

  /**
   * @brief Construct a new Collision Checker object
   *
   * @param robotShapeType    Type of the robot shape geometry
   * @param robotDimensions   Corresponding geometry dimensions
   * @param sensorPositionWRTbody         Position of the sensor w.r.t the robot
   * body - Considered constant
   * @param octreeRes         Resolution of the constructed OctTree
   */
  CollisionChecker(const ShapeType robot_shape_type,
                   const std::vector<float> &robot_dimensions,
                   const std::array<float, 3> &sensor_position_body,
                   const std::array<float, 4> &sensor_rotation_body,
                   const double octree_resolution = 0.01);

  /**
   * @brief Destroy the Collision Checker object
   *
   */
  ~CollisionChecker();

  /**
   * @brief Update the current state of the robot
   *
   * @param current_state
   */
  void updateState(const Path::State current_state);

  /**
   * @brief Update the sensor input from laser scan data
   *
   * @param ranges
   * @param angles
   */
  void updateScan(const std::vector<double> &ranges,
                  const std::vector<double> &angles);

  /**
   * @brief Update the sensor input from PointCloud (pcl::PointCloud) data
   *
   * @param cloud
   */
  void updatePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /**
   * @brief Update the sensor input from PointCloud like struct
   * std::vector<Control::Point3D>
   *
   * @param cloud
   */
  void updatePointCloud(const std::vector<Control::Point3D> &cloud);

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
   * @brief Get the Min Distance from given PointCloud data
   *
   * @param cloud
   * @return float
   */
  float getMinDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

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
  bool checkCollisions(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /**
   * @brief Check collisions between the robot and given PointCloud data
   *
   * @param cloud
   * @return true
   * @return false
   */
  bool checkCollisions(const std::vector<Control::Point3D> &cloud);

  /**
   * @brief Check collisions between the robot and given LaserScan data defined
   * by a vector of ranges values, minimum scan angle and scan angle step
   *
   * @param ranges
   * @param angle_min
   * @param angle_increment
   * @param height        Height of the constructed OctTree map, defaults to 0.1
   * @return true
   * @return false
   */
  bool checkCollisions(const std::vector<double> &ranges, double angle_min,
                       double angle_increment, double height = 0.1);

  /**
   * @brief Check collisions between the robot and given LaserScan data defined
   * by a vector of ranges values, and a vector of angles values
   *
   * @param ranges
   * @param angles
   * @param height        Height of the constructed OctTree map, defaults to 0.1
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

private:
  // Collision Manager
  fcl::DynamicAABBTreeCollisionManagerf *collManager;

  // Robot body geometry object
  std::shared_ptr<Body> body;
  double robotHeight_{1.0};
  std::shared_ptr<fcl::CollisionGeometryf> bodyGeometry;

  // Robot body collision object pointer
  fcl::CollisionObjectf *bodyObjPtr = nullptr;

  // Octree collision object pointers
  octomap::OcTree *octTree; // Octomap octree used to get data from laserscan or
                            // pointcloud and convert it to an Octreee
  fcl::OcTreef *tree =
      nullptr; // FCL Octree updated after coverting the Octomap octree
               // (required for creating the collision object)
  std::vector<fcl::CollisionObjectf *>
      OctreeBoxes; // Vector of Boxes collision objects used to check collisions
                   // with an octTree

  double octree_resolution_{0.01};

  // sensor tf with respect to the world
  Eigen::Isometry3f sensor_tf_world_ = Eigen::Isometry3f::Identity();

  Eigen::Isometry3f sensor_tf_body_ =
      Eigen::Isometry3f::Identity(); // Sensor transformation with
                                     // respect to the robot

  /**
   * @brief Updates the Octree collision object pointer
   *
   */
  void updateOctreePtr();

  /**
   * @brief Generates a vector of fcl::Box collision objects from an fcl::Octree
   *
   * @param boxes
   * @param tree
   */
  void generateBoxesFromOctomap(std::vector<fcl::CollisionObjectf *> &boxes,
                                fcl::OcTreef &tree);

  /**
   * @brief Helper method to convert PointCloud data to an Octomap
   *
   * @param cloud
   */
  void
  convertPointCloudToOctomap(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /**
   * @brief Helper method to convert PointCloud data to an Octomap
   *
   * @param cloud
   */
  void convertPointCloudToOctomap(const std::vector<Control::Point3D> &cloud);

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
