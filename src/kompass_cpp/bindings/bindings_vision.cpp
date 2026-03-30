#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "vision/depth_detector.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

namespace py = nanobind;
using namespace Kompass;

void bindings_vision(py::module_ &m) {
  auto m_vision = m.def_submodule("vision", "Vision and Detection module");

  py::class_<DepthDetector>(m_vision, "DepthDetector")
      // --- Constructor ---
      .def(
          "__init__",
          [](DepthDetector *t, const Eigen::Vector2f &depth_range,
             const Eigen::Vector3f &camera_in_body_translation,
             const Eigen::Vector4f &camera_in_body_rotation,
             const Eigen::Vector2f &focal_length,
             const Eigen::Vector2f &principal_point,
             const float depth_conversion_factor) {
            // Map Vector4f [x, y, z, w] to Eigen::Quaternionf (w, x, y, z)
            Eigen::Quaternionf quat(camera_in_body_rotation(3),  // w
                                    camera_in_body_rotation(0),  // x
                                    camera_in_body_rotation(1),  // y
                                    camera_in_body_rotation(2)); // z

            // Placement new to initialize the Python object
            new (t) DepthDetector(depth_range, camera_in_body_translation, quat,
                                  focal_length, principal_point,
                                  depth_conversion_factor);
          },
          py::arg("depth_range"), py::arg("camera_in_body_translation"),
          py::arg("camera_in_body_rotation"), py::arg("focal_length"),
          py::arg("principal_point"), py::arg("depth_conversion_factor") = 1e-3,
          "Initialize with camera translation and rotation (Vector4f as [x, y, "
          "z, w]).")

      // --- Converter Function ---
      .def(
          "compute_3d_detections",
          [](DepthDetector &self,
             const Eigen::MatrixX<unsigned short> &depth_img,
             const std::vector<Bbox2D> &input, float robot_x, float robot_y,
             float robot_yaw, float robot_speed) {
            Path::State state;
            state.x = robot_x;
            state.y = robot_y;
            state.yaw = robot_yaw;
            state.speed = robot_speed;

            self.updateBoxes(depth_img, input,
                             std::optional<Path::State>(state));
            return self.get3dDetections();
          },
          py::arg("depth_img"), py::arg("input"), py::arg("robot_x"),
          py::arg("robot_y"), py::arg("robot_yaw"), py::arg("robot_speed"))

      .def(
          "compute_3d_detections",
          [](DepthDetector &self,
             const Eigen::MatrixX<unsigned short> &depth_img,
             const std::vector<PointOfInterest> &input, float robot_x,
             float robot_y, float robot_yaw, float robot_speed) {
            Path::State state;
            state.x = robot_x;
            state.y = robot_y;
            state.yaw = robot_yaw;
            state.speed = robot_speed;

            self.updatePOIs(depth_img, input,
                            std::optional<Path::State>(state));
            return self.get3dDetections();
          },
          py::arg("depth_img"), py::arg("input"), py::arg("robot_x"),
          py::arg("robot_y"), py::arg("robot_yaw"), py::arg("robot_speed"));
}
