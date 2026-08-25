#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "vision/depth_detector.h"

#include "bindings.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

using namespace Kompass;

namespace {
// Private to file

inline Path::State makeState(const float x, const float y, const float yaw,
                             const float speed) {
  Path::State state;
  state.x = x;
  state.y = y;
  state.yaw = yaw;
  state.speed = speed;
  return state;
}

// The depth image accepts uint16 (mm) or float32 (m) arrays and the input is
// either 2D boxes or a PointsOfInterest set. The detection pass runs without
// the GIL
template <typename DepthArray, typename InputT>
std::vector<Bbox3D>
compute3dDetections(DepthDetector &self, const DepthArray &depth_img,
                    const InputT &input, float robot_x, float robot_y,
                    float robot_yaw, float robot_speed) {
  const auto view = toDepthView(depth_img);
  const auto state = makeState(robot_x, robot_y, robot_yaw, robot_speed);
  std::vector<Bbox3D> out;
  {
    py::gil_scoped_release release;
    if constexpr (std::is_same_v<InputT, PointsOfInterest>) {
      self.updatePOIs(view, input, state);
    } else {
      self.updateBoxes(view, input, state);
    }
    out = self.get3dDetections();
  }
  return out;
}

} // namespace

void bindings_vision(py::module_ &m) {
  auto m_vision = m.def_submodule("vision", "Vision and Detection module");

  py::enum_<DepthDetector::CameraFrameConvention>(m_vision,
                                                  "CameraFrameConvention")
      .value("OPTICAL", DepthDetector::CameraFrameConvention::Optical,
             "x right, y down, z into the image (REP 103 optical). What ROS "
             "Image/CameraInfo headers name and TF resolves.")
      .value("BODY_ALIGNED", DepthDetector::CameraFrameConvention::BodyAligned,
             "x forward, y left, z up (REP 103 body).");

  py::class_<DepthDetector>(m_vision, "DepthDetector")
      // --- Constructor ---
      .def(
          "__init__",
          [](DepthDetector *t, const Eigen::Vector2f &depth_range,
             const Eigen::Vector3f &camera_in_body_translation,
             const Eigen::Vector4f &camera_in_body_rotation,
             const Eigen::Vector2f &focal_length,
             const Eigen::Vector2f &principal_point,
             const float depth_conversion_factor,
             const DepthDetector::CameraFrameConvention convention) {
            // Map Vector4f [x, y, z, w] to Eigen::Quaternionf (w, x, y, z)
            Eigen::Quaternionf quat(camera_in_body_rotation(3),  // w
                                    camera_in_body_rotation(0),  // x
                                    camera_in_body_rotation(1),  // y
                                    camera_in_body_rotation(2)); // z

            // Placement new to initialize the Python object
            new (t) DepthDetector(depth_range, camera_in_body_translation, quat,
                                  focal_length, principal_point,
                                  depth_conversion_factor, convention);
          },
          py::arg("depth_range"), py::arg("camera_in_body_translation"),
          py::arg("camera_in_body_rotation"), py::arg("focal_length"),
          py::arg("principal_point"), py::arg("depth_conversion_factor") = 1e-3,
          py::arg("convention") = DepthDetector::CameraFrameConvention::Optical,
          "Initialize with camera translation and rotation (Vector4f as [x, y, "
          "z, w]). The pose is read in the optical convention by default, "
          "which is what a ROS TF lookup against an Image/CameraInfo frame_id "
          "gives; pass BODY_ALIGNED for a pose already in robot axes.")

      // --- Converter Functions ---
      // Zero-copy depth view (uint16 mm / float32 m, C-contiguous); one
      // typed overload per dtype x input kind, all sharing one template
      .def("compute_3d_detections",
           &compute3dDetections<DepthArrayU16, std::vector<Bbox2D>>,
           py::arg("depth_img"), py::arg("input"), py::arg("robot_x"),
           py::arg("robot_y"), py::arg("robot_yaw"), py::arg("robot_speed"))
      .def("compute_3d_detections",
           &compute3dDetections<DepthArrayF32, std::vector<Bbox2D>>,
           py::arg("depth_img"), py::arg("input"), py::arg("robot_x"),
           py::arg("robot_y"), py::arg("robot_yaw"), py::arg("robot_speed"))
      .def("compute_3d_detections",
           &compute3dDetections<DepthArrayU16, PointsOfInterest>,
           py::arg("depth_img"), py::arg("input"), py::arg("robot_x"),
           py::arg("robot_y"), py::arg("robot_yaw"), py::arg("robot_speed"))
      .def("compute_3d_detections",
           &compute3dDetections<DepthArrayF32, PointsOfInterest>,
           py::arg("depth_img"), py::arg("input"), py::arg("robot_x"),
           py::arg("robot_y"), py::arg("robot_yaw"), py::arg("robot_speed"));
}
