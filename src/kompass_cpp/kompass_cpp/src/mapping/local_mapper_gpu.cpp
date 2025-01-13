#include "mapping/local_mapper_gpu.h"
#include <sycl/sycl.hpp>

#ifndef NDEBUG
#define KERNEL_DEBUG(...)                                                      \
  namespace jit = sycl::AdaptiveCpp_jit;                                       \
  __acpp_if_target_sscp(                                                       \
      jit::compile_if(                                                         \
          jit::reflect<jit::reflection_query::compiler_backend>() ==           \
              jit::compiler_backend::host,                                     \
          [&]() { printf(__VA_ARGS__); }););
#else
#define KERNEL_DEBUG(...)
#endif

namespace Kompass {
namespace Mapping {

void LocalMapperGPU::scanToGrid(const std::vector<double> &angles,
                                const std::vector<double> &ranges,
                                Eigen::Ref<Eigen::MatrixXi> gridData) {

  try {

    m_q.fill(m_devicePtrGrid, static_cast<int>(OccupancyType::UNEXPLORED),
             m_gridHeight * m_gridWidth);

    m_q.memcpy(m_devicePtrAngles, angles.data(), sizeof(double) * m_scanSize);
    m_q.memcpy(m_devicePtrRanges, ranges.data(), sizeof(double) * m_scanSize);

    // command scope
    m_q.submit([&](sycl::handler &h) {
      // local copies of class members to be used inside the kernel
      const int rows = m_gridHeight;
      const int cols = m_gridWidth;
      const float resolution = m_resolution;
      const float laserscanOrientation = m_laserscanOrientation;

      auto devicePtrRanges = m_devicePtrRanges;
      auto devicePtrAngles = m_devicePtrAngles;
      auto devicePtrGrid = m_devicePtrGrid;

      sycl::range global_size(m_scanSize); // number of laser rays
      sycl::range work_group_size(
          m_maxPointsPerLine); // max num of cells per laser

      const sycl::vec<int, 2> v_centralPointLocal{m_centralPoint(0),
                                                  m_centralPoint(1)};
      const sycl::vec<float, 2> v_startPointLocal{m_laserscanPosition(0),
                                                  m_laserscanPosition(1)};

      const sycl::vec<int, 2> v_startPoint{m_startPoint(0), m_startPoint(1)};

      // local workgroup memory for storing destination point (x2, y2)
      auto toPoint = sycl::local_accessor<int, 1>{sycl::range{2}, h};

      // kernel scope
      h.parallel_for<class scanToGridKernel>(
          sycl::nd_range<1>{global_size * work_group_size, work_group_size},
          [=](sycl::nd_item<1> item) {
            const int group_id = item.get_group().get_group_id();
            const int local_id = item.get_local_id();

            double range = devicePtrRanges[group_id];
            double angle = devicePtrAngles[group_id];

            // calculate end point in the first thread of the block
            item.barrier(
                sycl::access::fence_space::local_space); // barrier within the
                                                         // group
            if (local_id == 0) {
              sycl::vec<float, 2> toPointLocal;
              toPointLocal[0] =
                  v_startPointLocal[0] +
                  (range * sycl::cos(laserscanOrientation + angle));
              toPointLocal[1] =
                  v_startPointLocal[1] +
                  (range * sycl::sin(laserscanOrientation + angle));

              toPoint[0] =
                  v_centralPointLocal[0] + (int)(toPointLocal[0] / resolution);
              toPoint[1] =
                  v_centralPointLocal[1] + (int)(toPointLocal[1] / resolution);
            }
            item.barrier(
                sycl::access::fence_space::local_space); // barrier within the
                                                         // group

            // calculate bressenham point
            int delta_x = toPoint[0] - v_startPoint[0];
            int delta_y = toPoint[1] - v_startPoint[1];

            float x_float, y_float;
            if (sycl::abs(delta_x) >= sycl::abs(delta_y)) {
              float g = (float)delta_y / (float)delta_x;
              x_float = v_startPoint[0] +
                        ((delta_x >= 0.0) ? 1 : ((delta_x < 0.0) ? -1 : 0)) *
                            local_id; // x1 + sign(delta_x) * j
              y_float = v_startPoint[1] + (g * (x_float - v_startPoint[0]));
            } else {
              float g = (float)delta_x / (float)delta_y;
              y_float = v_startPoint[1] +
                        ((delta_y > 0.0) ? 1 : ((delta_y < 0.0) ? -1 : 0)) *
                            local_id; // y1 + sign(delta_y) * j
              x_float = v_startPoint[0] + (g * (y_float - v_startPoint[1]));
            }
            int x(x_float);
            int y(y_float);

            // fill grid if applicable
            if (x >= 0 && x < rows && y >= 0 && y < cols) {

              // calculate distance
              sycl::vec<float, 2> destPointLocal{
                  (v_centralPointLocal[0] - x) * resolution,
                  (v_centralPointLocal[1] - y) * resolution};
              float distance =
                  sycl::distance(destPointLocal, v_startPointLocal);

              if (x == toPoint[0] && y == toPoint[1]) {
                devicePtrGrid[x * cols + y] = static_cast<int>(
                    Mapping::OccupancyType::OCCUPIED); // row major
              } else {
                if (distance < range)
                  devicePtrGrid[x * cols + y] = static_cast<int>(
                      Mapping::OccupancyType::EMPTY); // row major
              }
            }
          });
    });

    m_q.memcpy(m_out.data(), m_devicePtrGrid,
               sizeof(int) * m_gridWidth * m_gridHeight);

    m_q.wait_and_throw();

    // copy data to Eigen grid
    Eigen::Map<Eigen::MatrixXi>(m_out.data(), m_gridHeight, m_gridWidth)
        .swap(gridData);


  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
  }
}
} // namespace Mapping
} // namespace Kompass
