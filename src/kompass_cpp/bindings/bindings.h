#pragma once

#include "datatypes/span.h"
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace py = nanobind;

// Submodule binder entry points, one per bindings_*.cpp; called from
// NB_MODULE in bindings.cpp
void bindings_types(py::module_ &);
void bindings_config(py::module_ &);
void bindings_control(py::module_ &);
void bindings_mapping(py::module_ &);
void bindings_utils(py::module_ &);
void bindings_planning(py::module_ &);
void bindings_vision(py::module_ &);
void bindings_mapping_gpu(py::module_ &);
void bindings_utils_gpu(py::module_ &);

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

// Nx3 cartesian points (row-major so a numpy (N, 3) float32 array maps
// zero-copy); float64 or non-contiguous input converts with one copy
using RowMatrixX3f =
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
