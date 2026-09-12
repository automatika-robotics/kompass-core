#include "bindings.h"
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <vector>

#include "datatypes/control.h"
#include "datatypes/path.h"
#include "datatypes/tracking.h"
#include "datatypes/trajectory.h"
#include "utils/collision_check.h"
#include "utils/critical_zone_check.h"

using namespace Kompass;

std::string printControlCmd(const Control::Velocity2D &velocity_command) {
  return "{" + std::to_string(velocity_command.vx()) + ", " +
         std::to_string(velocity_command.vy()) + ", " +
         std::to_string(velocity_command.omega()) + "})";
}

// Types submodule
void bindings_types(py::module_ &m) {
  auto m_types = m.def_submodule("types", "KOMPASS CPP data types module");

  // Path types
  py::enum_<Path::InterpolationType>(m_types, "PathInterpolationType")
      .value("LINEAR", Path::InterpolationType::LINEAR)
      .value("CUBIC_SPLINE", Path::InterpolationType::CUBIC_SPLINE)
      .value("HERMITE_SPLINE", Path::InterpolationType::HERMITE_SPLINE)
      .export_values();

  py::class_<Path::State>(m_types, "State")
      .def(py::init<double, double, double, double>(), py::arg("x") = 0.0,
           py::arg("y") = 0.0, py::arg("yaw") = 0.0, py::arg("speed") = 0.0)
      .def_rw("x", &Path::State::x)
      .def_rw("y", &Path::State::y)
      .def_rw("yaw", &Path::State::yaw)
      .def_rw("speed", &Path::State::speed);

  py::class_<Path::PathPosition>(m_types, "PathPosition")
      .def(py::init<>())
      .def_rw("segment_index", &Path::PathPosition::segment_index)
      .def_rw("segment_length", &Path::PathPosition::segment_length)
      .def_rw("parallel_distance", &Path::PathPosition::parallel_distance)
      .def_rw("normal_distance", &Path::PathPosition::normal_distance);

  py::class_<Path::Path>(m_types, "Path")
      .def(py::init<const std::vector<Path::Point> &>(),
           py::arg("points") = std::vector<Path::Point>())
      .def(py::init<const Eigen::VectorXf &, const Eigen::VectorXf &,
                    const Eigen::VectorXf &>(),
           py::arg("x_points"), py::arg("y_points"), py::arg("z_points"))
      .def("reached_end", &Path::Path::endReached)
      .def("get_total_length", &Path::Path::totalPathLength)
      .def("size", &Path::Path::getSize)
      .def("getIndex", &Path::Path::getIndex, py::arg("index"))
      .def("x", &Path::Path::getX)
      .def("y", &Path::Path::getY);

  // Velocity control command
  py::class_<Control::Velocity2D>(m_types, "Velocity2D")
      .def(py::init<float, float, float, float>(), py::arg("vx") = 0.0,
           py::arg("vy") = 0.0, py::arg("omega") = 0.0,
           py::arg("steer_ang") = 0.0)
      .def_prop_rw("vx", &Control::Velocity2D::vx, &Control::Velocity2D::setVx)
      .def_prop_rw("vy", &Control::Velocity2D::vy, &Control::Velocity2D::setVy)
      .def_prop_rw("omega", &Control::Velocity2D::omega,
                   &Control::Velocity2D::setOmega)
      .def_prop_rw("steer_ang", &Control::Velocity2D::steer_ang,
                   &Control::Velocity2D::setSteerAng)
      .def("__str__", &printControlCmd);

  // Set of velocity control commands
  py::class_<Control::TrajectoryVelocities2D>(m_types, "TrajectoryVelocities2D")
      .def(py::init<>(), "Default constructor")
      // Maps Python 'length' (size of vector) to C++ 'numPointsPerTrajectory'
      // (length + 1)
      .def(
          "__init__",
          [](Control::TrajectoryVelocities2D *t, size_t length) {
            new (t) Control::TrajectoryVelocities2D(length + 1);
          },
          py::arg("length"),
          "Constructor taking the actual length of the velocity vectors")
      // Constructor from std::vector<Velocity2D>
      .def(py::init<const std::vector<Control::Velocity2D> &>(),
           "Initialize from a vector of Velocity2D", py::arg("velocities"))

      // Constructor from Eigen vectors
      .def(py::init<const Eigen::VectorXf &, const Eigen::VectorXf &,
                    const Eigen::VectorXf &>(),
           "Initialize from Eigen vectors", py::arg("vx"), py::arg("vy"),
           py::arg("omega"))
      .def_ro("vx", &Control::TrajectoryVelocities2D::vx,
              "Speed on x-axis (m/s)")
      .def_ro("vy", &Control::TrajectoryVelocities2D::vy,
              "Speed on y-axis (m/s)")
      .def_ro("omega", &Control::TrajectoryVelocities2D::omega,
              "Angular velocity (rad/s)")
      // Exposes 'length' as (numPointsPerTrajectory_ - 1)
      .def_prop_rw(
          "length",
          // Getter: return actual vector size
          [](const Control::TrajectoryVelocities2D &t) {
            return (t.numPointsPerTrajectory_ > 0)
                       ? (t.numPointsPerTrajectory_ - 1)
                       : 0;
          },
          // Setter: update internal counter
          [](Control::TrajectoryVelocities2D &t, size_t length) {
            t.numPointsPerTrajectory_ = length + 1;
          },
          "Actual length of the velocities vectors (mapped to "
          "numPointsPerTrajectory_ - 1)");

  py::class_<Control::TrajectoryPath>(m_types, "TrajectoryPath")
      .def(py::init<>())
      .def_ro("x", &Control::TrajectoryPath::x)
      .def_ro("y", &Control::TrajectoryPath::y)
      .def_ro("z", &Control::TrajectoryPath::z);

  py::class_<Control::Trajectory2D>(m_types, "Trajectory")
      .def(py::init<>())
      .def_ro("velocities", &Control::Trajectory2D::velocities)
      .def_ro("path", &Control::Trajectory2D::path);

  py::class_<Control::LaserScan>(m_types, "LaserScan")
      .def(
          "__init__",
          [](Control::LaserScan *self, Eigen::Ref<const Eigen::VectorXf> ranges,
             Eigen::Ref<const Eigen::VectorXf> angles) {
            new (self) Control::LaserScan(Eigen::VectorXf(ranges),
                                          Eigen::VectorXf(angles));
          },
          py::arg("ranges"), py::arg("angles"))
      .def_prop_ro(
          "ranges",
          [](const Control::LaserScan &s) -> Eigen::Ref<const Eigen::VectorXf> {
            return s.ranges;
          },
          py::rv_policy::reference_internal)
      .def_prop_ro(
          "angles",
          [](const Control::LaserScan &s) -> Eigen::Ref<const Eigen::VectorXf> {
            return s.angles;
          },
          py::rv_policy::reference_internal);

  // For collisions detection. The members mirror the FCL primitives the
  // collision checker can construct.
  py::enum_<CollisionChecker::ShapeType>(m_types, "RobotGeometry")
      .value("CYLINDER", CollisionChecker::ShapeType::CYLINDER)
      .value("BOX", CollisionChecker::ShapeType::BOX)
      .value("SPHERE", CollisionChecker::ShapeType::SPHERE)
      .value("ELLIPSOID", CollisionChecker::ShapeType::ELLIPSOID)
      .value("CAPSULE", CollisionChecker::ShapeType::CAPSULE)
      .value("CONE", CollisionChecker::ShapeType::CONE)
      .def_static("get",
                  [](const std::string &key) {
                    if (key == "CYLINDER")
                      return CollisionChecker::ShapeType::CYLINDER;
                    if (key == "BOX")
                      return CollisionChecker::ShapeType::BOX;
                    if (key == "SPHERE")
                      return CollisionChecker::ShapeType::SPHERE;
                    if (key == "ELLIPSOID")
                      return CollisionChecker::ShapeType::ELLIPSOID;
                    if (key == "CAPSULE")
                      return CollisionChecker::ShapeType::CAPSULE;
                    if (key == "CONE")
                      return CollisionChecker::ShapeType::CONE;
                    throw std::runtime_error("Invalid key");
                  })
      // Exposed so Python sizes the robot through the same derivation the
      // collision and critical zone checkers use
      .def_static("get_radius", &CollisionChecker::radiusOf,
                  py::arg("shape_type"), py::arg("dimensions"),
                  "Radius of the smallest circle in the xy-plane containing "
                  "the robot")
      .def_static("get_height", &CollisionChecker::heightOf,
                  py::arg("shape_type"), py::arg("dimensions"),
                  "Total extent of the robot along the z-axis")
      .def_static(
          "params_length",
          [](const CollisionChecker::ShapeType shape_type) {
            switch (shape_type) {
            case CollisionChecker::ShapeType::SPHERE:
              return 1;
            case CollisionChecker::ShapeType::CYLINDER:
            case CollisionChecker::ShapeType::CAPSULE:
            case CollisionChecker::ShapeType::CONE:
              return 2;
            case CollisionChecker::ShapeType::BOX:
            case CollisionChecker::ShapeType::ELLIPSOID:
              return 3;
            }
            throw std::runtime_error("Invalid robot geometry type");
          },
          py::arg("shape_type"), "Number of parameters the geometry requires");

  // For pointcloud data type
  py::enum_<PointFieldType>(m_types, "PointFieldType")
      .value("INT8", PointFieldType::INT8)
      .value("UINT8", PointFieldType::UINT8)
      .value("INT16", PointFieldType::INT16)
      .value("UINT16", PointFieldType::UINT16)
      .value("INT32", PointFieldType::INT32)
      .value("UINT32", PointFieldType::UINT32)
      .value("FLOAT32", PointFieldType::FLOAT32)
      .value("FLOAT64", PointFieldType::FLOAT64)
      .export_values()
      .def_static(
          "from_int",
          [](int value) {
            if (value < 1 || value > 8) {
              throw std::invalid_argument("Invalid integer for PointFieldType. "
                                          "Must be between 1 and 8.");
            }
            return static_cast<PointFieldType>(value);
          },
          py::arg("value"),
          "Creates a PointFieldType from its integer ID (1-8).");

  // Per-sensor mount configuration (used by mapper and critical zone checker)
  py::class_<SensorConfig>(m_types, "SensorConfig")
      .def(
          "__init__",
          [](SensorConfig *self, const Eigen::Vector3f &position,
             const Eigen::Vector4f &rotation,
             const PointFieldType cloud_field_type) {
            new (self) SensorConfig{position, rotation, cloud_field_type};
          },
          py::arg("position") = Eigen::Vector3f(0.0f, 0.0f, 0.0f),
          py::arg("rotation") = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f),
          py::arg("cloud_field_type") = PointFieldType::FLOAT32,
          "Sensor mount pose in the body frame (position + quaternion "
          "x, y, z, w) and the encoding of the sensor's point fields.")
      .def_rw("position", &SensorConfig::position)
      .def_rw("rotation", &SensorConfig::rotation)
      .def_rw("cloud_field_type", &SensorConfig::cloud_field_type)
      .def("__repr__", [](const SensorConfig &config) {
        return "SensorConfig(position=[" + std::to_string(config.position.x()) +
               ", " + std::to_string(config.position.y()) + ", " +
               std::to_string(config.position.z()) + "], rotation=[" +
               std::to_string(config.rotation.x()) + ", " +
               std::to_string(config.rotation.y()) + ", " +
               std::to_string(config.rotation.z()) + ", " +
               std::to_string(config.rotation.w()) + "], cloud_field_type=" +
               std::to_string(static_cast<int>(config.cloud_field_type)) + ")";
      });

  // For critical zone checking
  py::enum_<CriticalZoneChecker::InputType>(m_types, "SensorInputType")
      .value("LASERSCAN", CriticalZoneChecker::InputType::LASERSCAN)
      .value("POINTCLOUD", CriticalZoneChecker::InputType::POINTCLOUD)
      .def_static("get", [](const std::string &key) {
        if (key == "LASERSCAN")
          return CriticalZoneChecker::InputType::LASERSCAN;
        if (key == "POINTCLOUD")
          return CriticalZoneChecker::InputType::POINTCLOUD;
        throw std::runtime_error("Invalid key");
      });

  // Vision types
  py::class_<PointsOfInterest>(m_types, "PointsOfInterest")
      .def(py::init<>())
      .def(py::init<const PointsOfInterest &>())
      .def(
          py::init<const std::vector<Eigen::Vector2i> &,
                   const Eigen::Vector2i &, const float, const std::string &>(),
          py::arg("points"), py::arg("img_size") = Eigen::Vector2i(640, 480),
          py::arg("timestamp") = 0.0, py::arg("label") = "")
      .def_rw("points_2d", &PointsOfInterest::Points2D)
      .def_rw("timestamp", &PointsOfInterest::timestamp)
      .def_rw("label", &PointsOfInterest::label)
      .def_rw("img_size", &PointsOfInterest::img_size)
      .def_rw("vel", &PointsOfInterest::vel)
      .def("set_vel", &PointsOfInterest::setVel)
      .def("set_img_size", &PointsOfInterest::setImgSize);

  py::class_<Bbox2D>(m_types, "Bbox2D")
      .def(py::init<>())
      .def(py::init<const Bbox2D &>())
      .def(py::init<const Eigen::Vector2i &, const Eigen::Vector2i &,
                    const float, const std::string>(),
           py::arg("top_left_corner"), py::arg("size"),
           py::arg("timestamp") = 0.0, py::arg("label") = "")
      .def_rw("top_left_corner", &Bbox2D::top_corner)
      .def_rw("size", &Bbox2D::size)
      .def_rw("timestamp", &Bbox2D::timestamp)
      .def_rw("label", &Bbox2D::label)
      .def_rw("img_size", &Bbox2D::img_size)
      .def("set_vel", &Bbox2D::setVel)
      .def("set_img_size", &Bbox2D::setImgSize);

  py::class_<Bbox3D>(m_types, "Bbox3D")
      .def(py::init<>())
      .def(py::init<const Bbox3D &>())
      .def(py::init<const Eigen::Vector3f &, const Eigen::Vector3f &,
                    const Eigen::Vector2i &, const Eigen::Vector2i &,
                    const float, const std::string,
                    const std::vector<Eigen::Vector3f> &>(),
           py::arg("center"), py::arg("size"), py::arg("center_img_frame"),
           py::arg("size_img_frame"), py::arg("timestamp") = 0.0,
           py::arg("label") = "", py ::arg("pc_points") = py::list())
      .def_rw("center", &Bbox3D::center)
      .def_rw("size", &Bbox3D::size)
      .def_rw("center_img_frame", &Bbox3D::center_img_frame)
      .def_rw("size_img_frame", &Bbox3D::size_img_frame)
      .def_rw("pc_points", &Bbox3D::pc_points)
      .def_rw("timestamp", &Bbox3D::timestamp)
      .def_rw("label", &Bbox3D::label)
      .def_rw("sample_count", &Bbox3D::sample_count,
              "Number of in-range pixels of an aligned depth image or cloud "
              "points projected into the 2D box. Zero for a box not produced "
              "by the detector")
      .def_rw("source_index", &Bbox3D::source_index,
              "Index of the 2D box, or points-of-interest set, in the "
              "detector input this box was lifted from. -1 for a box "
              "not produced by the detector");
}
