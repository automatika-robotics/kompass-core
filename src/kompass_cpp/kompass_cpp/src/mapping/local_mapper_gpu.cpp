#include "mapping/local_mapper_gpu.h"
#include "utils/logger.h"
#include <hipSYCL/sycl/libkernel/builtins.hpp>
#include <sycl/sycl.hpp>

namespace Kompass {
namespace Mapping {

Eigen::MatrixXi &LocalMapperGPU::scanToGrid(const std::vector<double> &angles,
                                            const std::vector<double> &ranges) {

  try {

    m_q.fill(m_devicePtrGrid, static_cast<int>(OccupancyType::UNEXPLORED),
             m_gridHeight * m_gridWidth);

    m_q.memcpy(m_devicePtrAngles, angles.data(), sizeof(double) * m_scanSize);
    m_q.memcpy(m_devicePtrRanges, ranges.data(), sizeof(double) * m_scanSize);

    // command scope
    m_q.submit([&](sycl::handler &h) {
      // local copies of class members to be used inside the kernel
      const int rows = m_gridHeight; // implicitly casting size_t because its
                                     // not behaving nicely in the kernel
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
      auto deltas = sycl::local_accessor<int, 1>{sycl::range{2}, h};

      // kernel scope
      h.parallel_for<class scanToGridKernel>(
          sycl::nd_range<1>{global_size * work_group_size, work_group_size},
          [=](sycl::nd_item<1> item) {
            const size_t group_id = item.get_group().get_group_id();
            const size_t local_id = item.get_local_id();

            double range = devicePtrRanges[group_id];
            double angle = devicePtrAngles[group_id];

            // calculate end point and deltas in the first thread of the block
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
              deltas[0] = toPoint[0] - v_startPoint[0];
              deltas[1] = toPoint[1] - v_startPoint[1];
            }
            item.barrier(
                sycl::access::fence_space::local_space); // barrier within the
                                                         // group

            // calculate bressenham point
            float delta_x_f = static_cast<float>(deltas[0]);
            float delta_y_f = static_cast<float>(deltas[1]);
            float x_float, y_float;
            if (sycl::abs(deltas[0]) >= sycl::abs(deltas[1])) {
              float g = delta_y_f / delta_x_f;
              x_float =
                  v_startPoint[0] +
                  ((delta_x_f >= 0.0) ? 1 : ((delta_x_f < 0.0) ? -1 : 0)) *
                      local_id; // x1 + sign(delta_x) * j
              y_float = v_startPoint[1] + (g * (x_float - v_startPoint[0]));
            } else {
              float g = delta_x_f / delta_y_f;
              y_float = v_startPoint[1] +
                        ((delta_y_f > 0.0) ? 1 : ((delta_y_f < 0.0) ? -1 : 0)) *
                            local_id; // y1 + sign(delta_y) * j
              x_float = v_startPoint[0] + (g * (y_float - v_startPoint[1]));
            }

            int x = ceil(x_float);
            int y = ceil(y_float);

            // fill grid if applicable
            if (x >= 0 && x < rows && y >= 0 && y < cols) {

              // calculate distance
              sycl::vec<float, 2> destPointLocal{
                  (v_centralPointLocal[0] - x) * resolution,
                  (v_centralPointLocal[1] - y) * resolution};
              float distance =
                  sycl::distance(destPointLocal, v_startPointLocal);

              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  atomic_val(devicePtrGrid[x + y * rows]);
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  atomic_val_xstep(devicePtrGrid[(x - 1) + (y * rows)]);
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  atomic_val_ystep(devicePtrGrid[x + ((y - 1) * rows)]);
              if (x == toPoint[0] && y == toPoint[1]) {
                atomic_val.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::OCCUPIED));
              } else {
                if (distance < range)
                // Add super cover line points
                {
                  atomic_val.fetch_max(
                      static_cast<int>(Mapping::OccupancyType::EMPTY));
                  atomic_val_xstep.fetch_max(
                      static_cast<int>(Mapping::OccupancyType::EMPTY));
                  atomic_val_ystep.fetch_max(
                      static_cast<int>(Mapping::OccupancyType::EMPTY));
                }
              }
            }
          });
    });

    m_q.memcpy(gridData.data(), m_devicePtrGrid,
               sizeof(int) * m_gridWidth * m_gridHeight);

    m_q.wait_and_throw();

  } catch (const sycl::exception &e) {
    LOG_ERROR("Exception caught: ", e.what());
  }
  return gridData;
}
} // namespace Mapping
} // namespace Kompass
