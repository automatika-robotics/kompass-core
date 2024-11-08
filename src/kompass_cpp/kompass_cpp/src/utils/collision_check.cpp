#include "utils/collision_check.h"
#include "datatypes/trajectory.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <fcl/broadphase/default_broadphase_callbacks.h>
#include <fcl/common/types.h>
#include <fcl/narrowphase/collision_object.h>
#include <stdexcept>
#include <vector>

namespace Kompass {

CollisionChecker::CollisionChecker(
    const ShapeType robot_shape_type,
    const std::vector<float> &robot_dimensions,
    const std::array<float, 3> &sensor_position_body,
    const std::array<float, 4> &sensor_rotation_body,
    const double octree_resolution) {

  collManager = new fcl::DynamicAABBTreeCollisionManagerf();

  body = std::make_shared<Body>();

  octree_resolution_ = octree_resolution;
  octTree = new octomap::OcTree(octree_resolution_);

  body->shapeType = robot_shape_type;

  body->dimensions = robot_dimensions;

  // Construct  a geometry object based on the robot shape
  if (body->shapeType == ShapeType::CYLINDER) {
    bodyGeometry = std::make_shared<fcl::Cylinderf>(body->dimensions.at(0),
                                                    body->dimensions.at(1));
    robotHeight_ = body->dimensions.at(1);
  } else if (body->shapeType == ShapeType::BOX) {
    bodyGeometry = std::make_shared<fcl::Boxf>(
        body->dimensions.at(0), body->dimensions.at(1), body->dimensions.at(2));
    robotHeight_ = body->dimensions.at(2);
  } else {
    throw std::invalid_argument("Invalid robot geometry type");
  }

  // Set the body collision object pointer
  bodyObjPtr = new fcl::CollisionObjectf(bodyGeometry);

  // Init the sensor position w.r.t body
  sensor_tf_body_ = Control::getTransformation(
      Eigen::Quaternionf(sensor_rotation_body.data()),
      Eigen::Vector3f(sensor_position_body.data()));

  sensor_tf_world_ = sensor_tf_body_;
}

CollisionChecker::~CollisionChecker() {
  delete octTree;
  delete bodyObjPtr;
  delete collManager;
}

void CollisionChecker::generateBoxesFromOctomap(
    std::vector<fcl::CollisionObjectf *> &boxes, fcl::OcTreef &tree) {

  // Turn OctTree nodes into boxes
  std::vector<std::array<float, 6>> boxes_ = tree.toBoxes();

  // Clear old collision objects values
  boxes.clear();

  // Create collision objects from OctTree boxes
  for (std::size_t i = 0; i < boxes_.size(); ++i) {
    float x = boxes_[i][0];
    float y = boxes_[i][1];
    float z = boxes_[i][2];
    float size = boxes_[i][3];
    float cost = boxes_[i][4];
    float threshold = boxes_[i][5];

    fcl::Boxf *box = new fcl::Boxf(size, size, size);
    box->cost_density = cost;
    box->threshold_occupied = threshold;

    fcl::CollisionObjectf *obj =
        new fcl::CollisionObjectf(std::shared_ptr<fcl::CollisionGeometryf>(box),
                                  sensor_tf_world_.rotation(), // rotation
                                  sensor_tf_world_.translation() +
                                      Eigen::Vector3f(x, y, z)); // translation
    obj->computeAABB();
    boxes.push_back(obj);
  }

  LOG_DEBUG("Total Generated boxes above: ", boxes.size());
}

void CollisionChecker::updateOctreePtr() {
  // Create fcl::OcTree from octomap::OcTree
  tree = new fcl::OcTreef(std::shared_ptr<const octomap::OcTree>(octTree));

  // Transform the tree into a set of boxes and generate collision objects in
  // OctreeBoxes
  generateBoxesFromOctomap(OctreeBoxes, *tree);
}

void CollisionChecker::updateState(const Path::State current_state) {

  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation =
      Control::eulerToRotationMatrix(0.0, 0.0, current_state.yaw);

  body->tf = Control::getTransformation(
      rotation, Eigen::Vector3f(current_state.x, current_state.y, 0.0));

  bodyObjPtr->setTransform(body->tf);
  bodyObjPtr->computeAABB();
}

void CollisionChecker::updateScan(const std::vector<double> &ranges,
                                  const std::vector<double> &angles) {
  convertLaserScanToOctomap(ranges, angles, robotHeight_ / 2);
}

void CollisionChecker::updatePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  convertPointCloudToOctomap(cloud);
}

void CollisionChecker::updatePointCloud(
    const std::vector<Control::Point3D> &cloud) {
  convertPointCloudToOctomap(cloud);
}

void CollisionChecker::convertPointCloudToOctomap(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

  // Transform the sensor position to the world frame
  sensor_tf_world_ = sensor_tf_body_ * body->tf;

  // Clear old data
  octTree->clear();

  octomap::Pointcloud octomapCloud;
  for (const auto &point : cloud->points) {
    octomapCloud.push_back(point.x, point.y, point.z);
  }

  octTree->insertPointCloud(octomapCloud, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

void CollisionChecker::convertPointCloudToOctomap(
    const std::vector<Control::Point3D> &cloud) {

  // Transform the sensor position to the world frame
  // sensor_tf_world_ = sensor_tf_body_ * body->tf;
  sensor_tf_world_ = Eigen::Isometry3f::Identity();

  // Clear old data
  octTree->clear();

  octomap::Pointcloud octomapCloud;
  for (const auto &point : cloud) {
    octomapCloud.push_back(point.x, point.y, point.z);
  }

  octTree->insertPointCloud(octomapCloud, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

void CollisionChecker::convertLaserScanToOctomap(
    const std::vector<double> &ranges, double angle_min, double angle_increment,
    double height) {

  // Transform the sensor position to the world frame
  sensor_tf_world_ = sensor_tf_body_ * body->tf;

  // Clear old data
  octTree->clear();

  // Transform height to sensor frame
  float height_in_sensor = height - sensor_tf_body_.translation()[2];

  octomap::Pointcloud octomapCloud;
  for (size_t i = 0; i < ranges.size(); ++i) {
    float angle = angle_min + i * angle_increment;
    float x = ranges[i] * cos(angle);
    float y = ranges[i] * sin(angle);
    float z = height_in_sensor;

    octomapCloud.push_back(x, y, z);
  }

  octTree->insertPointCloud(octomapCloud, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

void CollisionChecker::convertLaserScanToOctomap(
    const std::vector<double> &ranges, const std::vector<double> &angles,
    double height) {

  // Transform the sensor position to the world frame
  sensor_tf_world_ = sensor_tf_body_ * body->tf;

  // Clear old data
  octTree->clear();

  // Transform height to sensor frame
  float height_in_sensor = height - sensor_tf_body_.translation()[2];

  octomap::Pointcloud octomapCloud;
  for (size_t i = 0; i < ranges.size(); ++i) {
    float x = ranges[i] * cos(angles[i]);
    float y = ranges[i] * sin(angles[i]);
    float z = height_in_sensor;
    octomapCloud.push_back(x, y, z);
  }
  octTree->insertPointCloud(octomapCloud, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

bool CollisionChecker::checkCollisionsOctree() {

  fcl::DefaultCollisionData<float> collisionData;

  collManager->clear();

  collManager->registerObjects(OctreeBoxes);

  collManager->setup();

  collManager->collide(bodyObjPtr, &collisionData,
                       fcl::DefaultCollisionFunction);

  return collisionData.result.isCollision();

  // NOTE: Code below for testing box by box
  // for (auto boxObj : OctreeBoxes) {
  //   collManager->clear();

  //   collManager->registerObject(boxObj);

  //   collManager->collide(bodyObjPtr, &collisionData,
  //                        fcl::DefaultCollisionFunction);

  //   collManager->setup();

  //   bool result = collisionData.result.isCollision();

  //   fcl::Vector3f trans = boxObj->getTranslation();
  //   fcl::Vector3f transRobot = bodyObjPtr->getTranslation();

  //   if (result) {
  //     fcl::Vector3f trans = boxObj->getTranslation();
  //     fcl::Vector3f transRobot = bodyObjPtr->getTranslation();
  //     LOG_DEBUG("Got collision with box at: {", trans[0], ", ", trans[1],
  //     ",",
  //               trans[2], "}. Robot at: {", transRobot[0], ", ",
  //               transRobot[1],
  //               ", ", transRobot[2], "}");

  //     return result;
  //   }
  // else{
  //   LOG_DEBUG("NO collision with box at: {", trans[0], ", ", trans[1], ",",
  //             trans[2], "}. Robot at: {", transRobot[0], ", ", transRobot[1],
  //             ", ", transRobot[2], "}");
  // }
  // }
  // return false;
}

float CollisionChecker::getMinDistance() {
  fcl::DefaultDistanceData<float> distanceData;

  collManager->clear();

  collManager->registerObjects(OctreeBoxes);

  collManager->setup();

  collManager->distance(bodyObjPtr, &distanceData,
                        fcl::DefaultDistanceFunction);

  return std::max<float>(0.0, distanceData.result.min_distance);
}

float CollisionChecker::getMinDistance(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  convertPointCloudToOctomap(cloud);
  return getMinDistance();
}

float CollisionChecker::getMinDistance(const std::vector<double> &ranges,
                                       double angle_min, double angle_increment,
                                       double height) {
  convertLaserScanToOctomap(ranges, angle_min, angle_increment, height);
  return getMinDistance();
}

float CollisionChecker::getMinDistance(const std::vector<double> &ranges,
                                       const std::vector<double> &angles,
                                       double height) {
  convertLaserScanToOctomap(ranges, angles, height);
  return getMinDistance();
}

bool CollisionChecker::checkCollisions(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  convertPointCloudToOctomap(cloud);
  return checkCollisionsOctree();
}

bool CollisionChecker::checkCollisions(const std::vector<double> &ranges,
                                       double angle_min, double angle_increment,
                                       double height) {
  convertLaserScanToOctomap(ranges, angle_min, angle_increment, height);
  return checkCollisionsOctree();
}

bool CollisionChecker::checkCollisions(const std::vector<double> &ranges,
                                       const std::vector<double> &angles,
                                       double height) {
  convertLaserScanToOctomap(ranges, angles, height);
  bool isCollision = checkCollisionsOctree();
  return isCollision;
}

bool CollisionChecker::checkCollisions() { return checkCollisionsOctree(); }

bool CollisionChecker::checkCollisions(const Path::State current_state) {
  auto m_stateObjPtr = new fcl::CollisionObjectf(bodyGeometry);
  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation =
      Control::eulerToRotationMatrix(0.0, 0.0, current_state.yaw);

  m_stateObjPtr->setTransform(Control::getTransformation(
      rotation, Eigen::Vector3f(current_state.x, current_state.y, 0.0)));
  m_stateObjPtr->computeAABB();

  // Setup a new collision manager and give it the state object
  fcl::DefaultCollisionData<float> collisionData;
  auto m_collManager = new fcl::DynamicAABBTreeCollisionManagerf();

  m_collManager->clear();
  m_collManager->registerObjects(OctreeBoxes);
  m_collManager->setup();
  m_collManager->collide(m_stateObjPtr, &collisionData,
                       fcl::DefaultCollisionFunction);

  // detele temp objects
  delete m_stateObjPtr;
  delete m_collManager;

  return collisionData.result.isCollision();
}

} // namespace Kompass
