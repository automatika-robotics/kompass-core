#pragma once

#include "datatypes/path.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace Kompass {

// Helper function to convert Euler angles to rotation matrix
inline Eigen::Matrix3f eulerToRotationMatrix(float roll, float pitch,
                                             float yaw) {
  // Implementation of conversion from Euler angles to a 3x3 rotation matrix
  Eigen::AngleAxisf rotZ(yaw, Eigen::Vector3f::UnitZ());
  Eigen::AngleAxisf rotY(pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rotX(roll, Eigen::Vector3f::UnitX());
  return (rotZ * rotY * rotX).matrix();
}

template <typename RotationType>
inline Eigen::Isometry3f
getTransformation(const RotationType &rotation_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal) {
  // Create a transformation matrix
  Eigen::Isometry3f transform_src_to_goal = Eigen::Isometry3f::Identity();

  // Set translation
  transform_src_to_goal.translate(translation_src_to_goal);

  // Set rotation based on the type of RotationType
  transform_src_to_goal.rotate(Eigen::Quaternionf(rotation_src_to_goal));

  return transform_src_to_goal;
}

inline Eigen::Isometry3f getTransformation(const Path::State state_in_frame) {
  Eigen::Matrix3f rotation_src_goal =
      eulerToRotationMatrix(0.0, 0.0, float(state_in_frame.yaw));
  Eigen::Vector3f translation_src_to_goal{float(state_in_frame.x),
                                          float(state_in_frame.y), 0.0};
  return getTransformation(rotation_src_goal, translation_src_to_goal);
}
/**
 * @brief Transform a position given translation vector and rotation quat or
 * matrix
 *
 * @param object_pose_in_frame_src
 * @param rotation_src_to_goal
 * @param translation_src_to_goal
 * @return Eigen::Vector3f
 */

template <typename RotationType>
inline Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const RotationType &rotation_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal) {
  // Construct the transformation matrix
  Eigen::Isometry3f transform_src_to_goal =
      getTransformation(rotation_src_to_goal, translation_src_to_goal);

  // Transform the object pose from source frame to goal frame
  return transform_src_to_goal * object_pose_in_frame_src;
}

inline Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Isometry3f &transform_src_to_goal) {

  // Transform the object pose from source frame to goal frame
  return transform_src_to_goal * object_pose_in_frame_src;
}

/**
 * @brief Transform a pose given translation vector and rotation quaternion or
 * matrix
 *
 * @param object_pose_in_frame_src
 * @param rotation_src_to_goal
 * @param translation_src_to_goal
 * @return Eigen::Isometry3f
 */

template <typename RotationType>
inline Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const RotationType &rotation_src_to_goal,
              const Eigen::Vector3f &translation_src_to_goal) {

  // Construct the transformation matrix
  Eigen::Isometry3f transform_src_to_goal =
      getTransformation(rotation_src_to_goal, translation_src_to_goal);

  Eigen::Isometry3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

inline Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Isometry3f &transform_src_to_goal) {
  return transform_src_to_goal * object_pose_in_frame_src;
}

} // namespace Kompass
