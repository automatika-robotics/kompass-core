#pragma once

#include "datatypes/path.h"
#include "datatypes/sensors.h"
#include "datatypes/span.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = nanobind;

// ---------------------------------------------------------------------------
// Submodule binder entry points, one per bindings_*.cpp; called from
// NB_MODULE in bindings.cpp
// ---------------------------------------------------------------------------
void bindings_types(py::module_ &);
void bindings_config(py::module_ &);
void bindings_control(py::module_ &);
void bindings_mapping(py::module_ &);
void bindings_utils(py::module_ &);
void bindings_planning(py::module_ &);
void bindings_vision(py::module_ &);
void bindings_mapping_gpu(py::module_ &);
void bindings_utils_gpu(py::module_ &);

// ---------------------------------------------------------------------------
// PointCloud2-style byte buffers
// ---------------------------------------------------------------------------

// Zero-copy input type for PointCloud2-style byte buffers: accepts numpy
// uint8 arrays (including read-only np.frombuffer views) and Python
// bytes/bytearray/memoryview via the buffer protocol. Wrong-dtype or
// non-contiguous numpy input converts with a single C-level copy. Lists are
// rejected.
using ByteArray =
    py::ndarray<const uint8_t, py::ndim<1>, py::c_contig, py::device::cpu>;

inline Kompass::ByteSpan toSpan(const ByteArray &a) {
  return Kompass::ByteSpan(a.data(), a.size());
}

// ---------------------------------------------------------------------------
// Aligned depth images
// ---------------------------------------------------------------------------

// Zero-copy input types for aligned depth images: a 2-D C-contiguous
// (rows, cols) numpy array, uint16 (depth in millimetres, ROS 16UC1) or
// float32 (depth in metres, ROS 32FC1). Bound with noconvert so a wrong dtype
// or a non-contiguous layout raises instead of being copied into the other
// dtype's overload.
using DepthArrayU16 =
    py::ndarray<const uint16_t, py::ndim<2>, py::c_contig, py::device::cpu>;
using DepthArrayF32 =
    py::ndarray<const float, py::ndim<2>, py::c_contig, py::device::cpu>;

// The returned view aliases the array's buffer: the array handle must
// outlive the C++ call the view is passed to. Must be called with the GIL
// held.
inline Kompass::DepthImageView toDepthView(const DepthArrayU16 &a) {
  return Kompass::DepthImageView(
      Kompass::ByteSpan(reinterpret_cast<const uint8_t *>(a.data()),
                        a.size() * sizeof(uint16_t)),
      static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
      PointFieldType::UINT16);
}

inline Kompass::DepthImageView toDepthView(const DepthArrayF32 &a) {
  return Kompass::DepthImageView(
      Kompass::ByteSpan(reinterpret_cast<const uint8_t *>(a.data()),
                        a.size() * sizeof(float)),
      static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
      PointFieldType::FLOAT32);
}

// ---------------------------------------------------------------------------
// Zero-Copy N,3 setup (mostly for world-frame Pointcloud (pc.xyz) consumers)
// ---------------------------------------------------------------------------

// Nx3 cartesian points (row-major so a numpy (N, 3) float32 array maps
// zero-copy); float64 or non-contiguous input converts with one copy
using RowMatrixX3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

// The Nx3 float32 buffer is reinterpreted as a Span of points with NO
// per-point build, which requires Path::Point (Eigen::Vector3f) to stay 3
// packed floats
static_assert(sizeof(Path::Point) == 3 * sizeof(float),
              "Path::Point must be 3 packed floats for the zero-copy "
              "Span reinterpret in the cloud bindings");

// The returned span aliases the Ref's buffer: the Ref (and the numpy array
// behind it) must outlive the C++ call the span is passed to.
inline Kompass::Span<Path::Point>
toPointSpan(const Eigen::Ref<const RowMatrixX3f> &cloud) {
  return Kompass::Span<Path::Point>(
      reinterpret_cast<const Path::Point *>(cloud.data()),
      static_cast<size_t>(cloud.rows()));
}

// ---------------------------------------------------------------------------
// Multi-sensor PointCloud2 batches (dict elements -> PointCloudViews)
// ---------------------------------------------------------------------------

// Extracts one PointCloudView from a Python cloud element. An element is a dict
// (extra keys are ignored), or None, which yields an empty view. The element's
// byte buffer crosses zero-copy. Its owner handle is appended to
// `keepalive`, which must outlive the C++ call the views are passed to. Must be
// called with the GIL held.
inline Kompass::PointCloudView
extractCloudView(py::handle element, const size_t index,
                 std::vector<ByteArray> &keepalive) {
  if (!element.is_valid() || element.is_none()) {
    return Kompass::PointCloudView{};
  }
  auto fail = [index](const std::string &message) {
    throw std::invalid_argument("clouds[" + std::to_string(index) +
                                "]: " + message);
  };
  if (!py::isinstance<py::dict>(element)) {
    fail("expected a dict (e.g. PointCloudData.asdict()) or None");
  }
  py::dict cloud = py::borrow<py::dict>(element);

  auto field = [&](const char *name) -> py::object {
    if (!cloud.contains(name)) {
      fail("missing field '" + std::string(name) + "'");
    }
    return cloud[name];
  };
  auto int_field = [&](const char *name) -> int {
    int value = 0;
    if (!py::try_cast<int>(field(name), value)) {
      fail("field '" + std::string(name) + "' must be an integer");
    }
    return value;
  };

  ByteArray data;
  if (!py::try_cast<ByteArray>(field("data"), data)) {
    fail("'data' must be a contiguous uint8 buffer (numpy uint8 array or "
         "bytes)");
  }

  Kompass::PointCloudView view;
  view.point_step = int_field("point_step");
  view.row_step = int_field("row_step");
  view.height = int_field("height");
  view.width = int_field("width");
  view.x_offset = int_field("x_offset");
  view.y_offset = int_field("y_offset");
  view.z_offset = int_field("z_offset");
  keepalive.push_back(data);
  view.data = toSpan(keepalive.back());
  return view;
}

// Extracts a whole batch. clouds[i] pairs with the mapper/checker's i-th
// configured sensor (positional). Must be called with the GIL held; the
// returned views stay valid for as long as `keepalive` lives
inline std::vector<Kompass::PointCloudView>
extractCloudViews(py::sequence clouds, std::vector<ByteArray> &keepalive) {
  const size_t count = py::len(clouds);
  std::vector<Kompass::PointCloudView> views;
  views.reserve(count);
  keepalive.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    views.push_back(extractCloudView(clouds[i], i, keepalive));
  }
  return views;
}
