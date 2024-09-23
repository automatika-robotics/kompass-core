#pragma once

#include "datatypes/path.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Geometry/Transform.h>

namespace Kompass {

namespace Control {

Eigen::Isometry3f getTransformation(const Path::State state_in_frame);

Eigen::Isometry3f
getTransformation(const Eigen::Quaternionf &quat_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal);

Eigen::Isometry3f
getTransformation(const Eigen::Matrix3f &rotation_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal);

/**
 * @brief Transform a position given translation vector and rotation
 * quaternion
 *
 * @param object_pose_in_frame_src
 * @param quat_src_to_goal
 * @param translation_src_to_goal
 * @return Eigen::Vector3f
 */
Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Quaternionf &quat_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal);

/**
 * @brief Transform a position given translation vector and rotation matrix
 *
 * @param object_pose_in_frame_src
 * @param rotation_src_to_goal
 * @param translation_src_to_goal
 * @return Eigen::Vector3f
 */

Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Matrix3f &rotation_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal);

Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Isometry3f &transform_src_to_goal);

/**
 * @brief Transform a pose given translation vector and rotation quaternion
 *
 * @param object_pose_in_frame_src
 * @param quat_src_to_goal
 * @param translation_src_to_goal
 * @return Eigen::Isometry3f
 */

Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Quaternionf &quat_src_to_goal,
              const Eigen::Vector3f &translation_src_to_goal);

/**
 * @brief Transform a pose given translation vector and rotation matrix
 *
 * @param object_pose_in_frame_src
 * @param rotation_src_to_goal
 * @param translation_src_to_goal
 * @return Eigen::Isometry3f
 */

Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Matrix3f &rotation_src_to_goal,
              const Eigen::Vector3f &translation_src_to_goal);

Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Isometry3f &transform_src_to_goal);

Eigen::Matrix3f eulerToRotationMatrix(float eulerX, float eulerY, float eulerZ);

} // namespace Control
} // namespace Kompass
