#include "mapping/local_mapper_gpu.h"
#include "utils/logger.h"
#include "utils/pointcloud.h"
#include <sycl/sycl.hpp>

namespace Kompass {
namespace Mapping {

Eigen::MatrixXi &LocalMapperGPU::scanToGrid(const std::vector<int8_t> &data,
                                            int point_step, int row_step,
                                            int height, int width,
                                            float x_offset, float y_offset,
                                            float z_offset) {
  pointCloudToLaserScanFromRaw(
      data, point_step, row_step, height, width, x_offset, y_offset, z_offset,
      m_rangeMax, m_minHeight, m_maxHeight, m_scanSize, initializedRanges);
  return scanToGrid(initializedAngles, initializedRanges);
}

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
      auto devicePtrDistances = m_devicePtrDistances;

      sycl::range global_size(m_scanSize); // number of laser rays
      sycl::range work_group_size(
          m_maxPointsPerLine); // max num of cells per laser

      const sycl::vec<int, 2> v_centralPoint{m_centralPoint(0),
                                             m_centralPoint(1)};
      const sycl::vec<float, 2> v_startPointLocal{m_laserscanPosition(0),
                                                  m_laserscanPosition(1)};

      const sycl::vec<int, 2> v_startPoint{m_startPoint(0), m_startPoint(1)};

      // local workgroup memory for storing destination point (x2, y2), deltas
      // and steps
      auto toPoint = sycl::local_accessor<int, 1>{sycl::range{2}, h};
      auto deltas = sycl::local_accessor<int, 1>{sycl::range{2}, h};
      auto steps = sycl::local_accessor<int, 1>{sycl::range{2}, h};

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
                  v_centralPoint[0] + ceil(toPointLocal[0] / resolution);
              toPoint[1] =
                  v_centralPoint[1] + ceil(toPointLocal[1] / resolution);
              deltas[0] = toPoint[0] - v_startPoint[0];
              deltas[1] = toPoint[1] - v_startPoint[1];
              steps[0] = (deltas[0] >= 0) ? 1 : -1;
              steps[1] = (deltas[1] >= 0) ? 1 : -1;
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

            int x = round(x_float);
            int y = round(y_float);

            // fill grid if applicable
            if (x >= 0 && x < rows && y >= 0 && y < cols) {
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  atomic_val(devicePtrGrid[x + y * rows]);
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  atomic_val_xstep(devicePtrGrid[(x - steps[0]) + (y * rows)]);
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::local_space>
                  atomic_val_ystep(devicePtrGrid[x + ((y - steps[1]) * rows)]);
              if (x == toPoint[0] && y == toPoint[1]) {
                // Add obstacle point while taking care of inside corners
                atomic_val.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::OCCUPIED));
                atomic_val_xstep.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
                atomic_val_ystep.fetch_max(
                    static_cast<int>(Mapping::OccupancyType::EMPTY));
              } else {
                if (devicePtrDistances[x + y * rows] < range)
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

  } catch (sycl::exception const &e) {
    LOG_ERROR("SYCL exception caught: ", e.what());
    throw; // Re-throw to Python
  } catch (std::exception const &e) {
    LOG_ERROR("Standard exception caught: ", e.what());
    throw; // Re-throw to Python
  }
  return gridData;
}
} // namespace Mapping
} // namespace Kompass
