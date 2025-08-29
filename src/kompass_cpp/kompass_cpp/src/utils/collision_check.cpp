#include "utils/collision_check.h"
#include "utils/logger.h"
#include "utils/transformation.h"
#include <algorithm>
#include <cmath>
#include <fcl/broadphase/default_broadphase_callbacks.h>
#include <fcl/common/types.h>
#include <fcl/geometry/octree/octree.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/narrowphase/collision_object.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace Kompass {

CollisionChecker::CollisionChecker(
    const ShapeType robot_shape_type,
    const std::vector<float> &robot_dimensions,
    const Eigen::Vector3f &sensor_position_body,
    const Eigen::Quaternionf &sensor_rotation_body,
    const double octree_resolution) {

  collManager_ = std::make_unique<fcl::DynamicAABBTreeCollisionManagerf>();

  body = std::make_shared<Body>();

  octree_resolution_ = octree_resolution;
  octTree_ = std::make_shared<octomap::OcTree>(octree_resolution_);
  fclTree_ = std::make_shared<fcl::OcTreef>(octTree_);

  body->shapeType = robot_shape_type;

  body->dimensions = robot_dimensions;

  // Construct  a geometry object based on the robot shape
  if (body->shapeType == ShapeType::CYLINDER) {
    bodyGeometry_ = std::make_shared<fcl::Cylinderf>(body->dimensions.at(0),
                                                     body->dimensions.at(1));

    robotHeight_ = body->dimensions.at(1);
    robotRadius_ = body->dimensions.at(0);
  } else if (body->shapeType == ShapeType::BOX) {
    bodyGeometry_ = std::make_shared<fcl::Boxf>(
        body->dimensions.at(0), body->dimensions.at(1), body->dimensions.at(2));
    robotHeight_ = body->dimensions.at(2);
    robotRadius_ = std::sqrt(pow(body->dimensions.at(0), 2) +
                             pow(body->dimensions.at(1), 2)) /
                   2;
  } else if (body->shapeType == ShapeType::SPHERE) {
    bodyGeometry_ = std::make_shared<fcl::Spheref>(body->dimensions.at(0));
    robotRadius_ = body->dimensions.at(0);
    robotHeight_ = 2 * body->dimensions.at(0);
    ;
  } else {
    throw std::invalid_argument("Invalid robot geometry type");
  }

  // Set the body collision object pointer
  bodyObjPtr_ = std::make_shared<fcl::CollisionObjectf>(bodyGeometry_);

  // Init the sensor position w.r.t body
  sensor_tf_body_ =
      getTransformation(sensor_rotation_body, sensor_position_body);

  sensor_tf_world_ = sensor_tf_body_;
}

void CollisionChecker::resetOctreeResolution(const double resolution) {
  if (resolution != octree_resolution_) {
    octree_resolution_ = resolution;
    octTree_->setResolution(octree_resolution_);
  }
}

float CollisionChecker::getRadius() const { return robotRadius_; }

std::vector<fcl::CollisionObjectf *>
CollisionChecker::generateBoxesFromOctomap(fcl::OcTreef &tree) {
  std::vector<fcl::CollisionObjectf *> boxes;
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
    // Uncomment for debug
    // auto trans = obj->getTransform();
    // std::cout << "Adding box with translation " << trans.translation() <<
    // std::endl;
    boxes.push_back(obj);
  }

  LOG_DEBUG("Total Generated boxes above: ", boxes.size());
  return boxes;
}

void CollisionChecker::updateOctreePtr() {
  fclTree_.reset(new fcl::OcTreef(octTree_));
  OctreeCollObj_ =
      std::make_unique<fcl::CollisionObjectf>(fclTree_, sensor_tf_world_);
  OctreeCollObj_->computeAABB();
}

void CollisionChecker::updateState(const Path::State current_state) {

  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation = eulerToRotationMatrix(0.0, 0.0, current_state.yaw);

  body->tf = getTransformation(
      rotation, Eigen::Vector3f(current_state.x, current_state.y, 0.0));

  bodyObjPtr_->setTransform(body->tf);
  bodyObjPtr_->computeAABB();
}

void CollisionChecker::updateState(const double x, const double y,
                                   const double yaw) {

  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation = eulerToRotationMatrix(0.0, 0.0, yaw);

  body->tf = getTransformation(rotation, Eigen::Vector3f(x, y, 0.0));

  bodyObjPtr_->setTransform(body->tf);
  bodyObjPtr_->computeAABB();
}

void CollisionChecker::updateScan(const std::vector<double> &ranges,
                                  const std::vector<double> &angles) {
  convertLaserScanToOctomap(ranges, angles, robotHeight_ / 2);
}

void CollisionChecker::updatePointCloud(const std::vector<Path::Point> &cloud,
                                        const bool global_frame) {
  convertPointCloudToOctomap(cloud);
}

void CollisionChecker::update3DMap(const std::vector<Eigen::Vector3f> &points,
                                   const bool global_frame) {
  if (global_frame) {
    sensor_tf_world_ = Eigen::Isometry3f::Identity();
  } else {
    // Transform the sensor position to the world frame
    sensor_tf_world_ = body->tf * sensor_tf_body_;
  }

  // Clear old data
  octTree_->clear();

  octomapCloud_.clear();
  for (auto &point : points) {
    octomapCloud_.push_back(point.x(), point.y(), point.z());
  }

  octTree_->insertPointCloud(octomapCloud_, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

void CollisionChecker::convertPointCloudToOctomap(
    const std::vector<Path::Point> &cloud, const bool global_frame) {

  // Transform the sensor position to the world frame
  // NOTE: Transformation will be applied to the points when creating the
  // collision object
  if (global_frame) {
    sensor_tf_world_ = Eigen::Isometry3f::Identity();
  } else {
    // Transform the sensor position to the world frame
    sensor_tf_world_ = body->tf * sensor_tf_body_;
  }

  // Clear old data
  octTree_->clear();

  octomapCloud_.clear();
  for (const auto &point : cloud) {
    octomapCloud_.push_back(point.x(), point.y(), point.z());
  }

  octTree_->insertPointCloud(octomapCloud_, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

void CollisionChecker::convertLaserScanToOctomap(
    const std::vector<double> &ranges, double angle_min, double angle_increment,
    double height) {

  // Transform the sensor position to the world frame
  // NOTE: Transformation will be applied to the points when creating the
  // collision object
  sensor_tf_world_ = body->tf * sensor_tf_body_;

  // Clear old data
  octTree_->clear();

  // Transform height to sensor frame
  float height_in_sensor = -sensor_tf_body_.translation()[2] / 2;
  float x, y, z, angle;
  octomapCloud_.clear();

  for (size_t i = 0; i < ranges.size(); ++i) {
    angle = angle_min + i * angle_increment;
    x = ranges[i] * cos(angle);
    y = ranges[i] * sin(angle);
    z = height_in_sensor;
    octomapCloud_.push_back(x, y, z);
  }

  octTree_->insertPointCloud(octomapCloud_, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

void CollisionChecker::convertLaserScanToOctomap(
    const std::vector<double> &ranges, const std::vector<double> &angles,
    double height) {

  // Transform the sensor position to the world frame
  // NOTE: Transformation will be applied to the points when creating the
  // collision object
  sensor_tf_world_ = body->tf * sensor_tf_body_;

  // Clear old data
  octTree_->clear();

  // Transform height to sensor frame
  float height_in_sensor = -sensor_tf_body_.translation()[2] / 2;

  float x, y, z;
  octomapCloud_.clear();

  for (size_t i = 0; i < ranges.size(); ++i) {
    x = ranges[i] * cos(angles[i]);
    y = ranges[i] * sin(angles[i]);
    z = height_in_sensor;
    octomapCloud_.push_back(x, y, z);
  }
  octTree_->insertPointCloud(octomapCloud_, octomap::point3d(0, 0, 0));

  updateOctreePtr();
}

bool CollisionChecker::checkCollisionsOctree() {

  fcl::DefaultCollisionData<float> collisionData;

  collManager_->clear();

  collManager_->registerObject(OctreeCollObj_.get());

  collManager_->setup();

  collManager_->collide(bodyObjPtr_.get(), &collisionData,
                        fcl::DefaultCollisionFunction);

  return collisionData.result.isCollision();

  // NOTE: Code below for testing box by box
  // auto OctreeBoxes = generateBoxesFromOctomap(*fclTree_);
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

  collManager_->clear();

  collManager_->registerObject(OctreeCollObj_.get());

  collManager_->setup();

  collManager_->distance(bodyObjPtr_.get(), &distanceData,
                         fcl::DefaultDistanceFunction);

  return std::max<float>(0.0, distanceData.result.min_distance);
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
  auto m_stateObjPtr = std::make_unique<fcl::CollisionObjectf>(bodyGeometry_);
  // Get the body tf matrix from euler angles / new position
  Eigen::Matrix3f rotation = eulerToRotationMatrix(0.0, 0.0, current_state.yaw);

  m_stateObjPtr->setTransform(getTransformation(
      rotation, Eigen::Vector3f(current_state.x, current_state.y, 0.0)));
  m_stateObjPtr->computeAABB();

  // Setup a new collision manager and give it the state object
  fcl::DefaultCollisionData<float> collisionData;
  auto m_collManager =
      std::make_unique<fcl::DynamicAABBTreeCollisionManagerf>();

  m_collManager->clear();
  m_collManager->registerObject(OctreeCollObj_.get());
  m_collManager->setup();
  m_collManager->collide(m_stateObjPtr.get(), &collisionData,
                         fcl::DefaultCollisionFunction);

  return collisionData.result.isCollision();
}

} // namespace Kompass
