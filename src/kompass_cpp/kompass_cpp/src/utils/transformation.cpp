#include "utils/transformation.h"

namespace Kompass {

namespace Control {

Eigen::Isometry3f getTransformation(const Path::State state_in_frame) {
  Eigen::Matrix3f rotation_src_goal =
      eulerToRotationMatrix(0.0, 0.0, float(state_in_frame.yaw));
  Eigen::Vector3f translation_src_to_goal{float(state_in_frame.x),
                                          float(state_in_frame.y), 0.0};
  return getTransformation(rotation_src_goal, translation_src_to_goal);
}

Eigen::Isometry3f
getTransformation(const Eigen::Quaternionf &quat_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal) {
  // Create a transformation matrix
  Eigen::Isometry3f transform_src_to_goal = Eigen::Isometry3f::Identity();

  // Set rotation (from quaternion)
  transform_src_to_goal.rotate(quat_src_to_goal);

  // Set translation
  transform_src_to_goal.pretranslate(translation_src_to_goal);

  return transform_src_to_goal;
}

Eigen::Isometry3f
getTransformation(const Eigen::Matrix3f &rotation_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal) {
  // Create a transformation matrix
  Eigen::Isometry3f transform_src_to_goal = Eigen::Isometry3f::Identity();

  // Set rotation (from rotation matrix)
  transform_src_to_goal.rotate(rotation_src_to_goal);

  // Set translation
  transform_src_to_goal.pretranslate(translation_src_to_goal);

  return transform_src_to_goal;
}

Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Isometry3f &transform_src_to_goal) {

  // Transform the object pose from source frame to goal frame
  Eigen::Vector3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Quaternionf &quat_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal) {
  // Construct the transformation matrix
  Eigen::Isometry3f transform_src_to_goal =
      getTransformation(quat_src_to_goal, translation_src_to_goal);

  // Transform the object pose from source frame to goal frame
  Eigen::Vector3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

Eigen::Vector3f
transformPosition(const Eigen::Vector3f &object_pose_in_frame_src,
                  const Eigen::Matrix3f &rotation_src_to_goal,
                  const Eigen::Vector3f &translation_src_to_goal) {
  // Construct the transformation matrix
  Eigen::Isometry3f transform_src_to_goal =
      getTransformation(rotation_src_to_goal, translation_src_to_goal);

  Eigen::Vector3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Quaternionf &quat_src_to_goal,
              const Eigen::Vector3f &translation_src_to_goal) {

  // Construct the transformation matrix
  Eigen::Isometry3f transform_src_to_goal =
      getTransformation(quat_src_to_goal, translation_src_to_goal);

  Eigen::Isometry3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Matrix3f &rotation_src_to_goal,
              const Eigen::Vector3f &translation_src_to_goal) {

  // Construct the transformation matrix
  Eigen::Isometry3f transform_src_to_goal =
      getTransformation(rotation_src_to_goal, translation_src_to_goal);

  Eigen::Isometry3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

Eigen::Isometry3f
transformPose(const Eigen::Isometry3f &object_pose_in_frame_src,
              const Eigen::Isometry3f &transform_src_to_goal) {
  Eigen::Isometry3f object_pose_in_goal =
      transform_src_to_goal * object_pose_in_frame_src;

  return object_pose_in_goal;
}

Eigen::Matrix3f eulerToRotationMatrix(float eulerX, float eulerY,
                                      float eulerZ) {
  Eigen::Matrix3f R(3, 3);

  // Matrix for single rotation around x-axis with angle eulerX (Roll)
  Eigen::Matrix3f R_x{{1, 0, 0},
                      {0, std::cos(eulerX), -std::sin(eulerX)},
                      {0, std::sin(eulerX), std::cos(eulerX)}};

  // Matrix for single rotation around y-axis with angle eulerY (Pitch)
  Eigen::Matrix3f R_y{{std::cos(eulerY), 0, std::sin(eulerY)},
                      {0, 1, 0},
                      {-std::sin(eulerY), 0, std::cos(eulerY)}};

  // Matrix for single rotation around z-axis with angle eulerZ (Yaw)
  Eigen::Matrix3f R_z{{std::cos(eulerZ), -std::sin(eulerZ), 0},
                      {std::sin(eulerZ), std::cos(eulerZ), 0},
                      {0, 0, 1}};

  R = R_z * R_y * R_x;

  return R;
}

} // namespace Control
} // namespace Kompass
