#pragma once

#include "local_mapper.h"
#include "utils/logger.h"
#include <Eigen/Dense>
#include <sycl/sycl.hpp>

namespace Kompass {
namespace Mapping {

class LocalMapperGPU : public LocalMapper {
public:
  // Constructor
  LocalMapperGPU(const int gridHeight, const int gridWidth, float resolution,
                 const Eigen::Vector3f &laserscanPosition,
                 float laserscanOrientation, int scanSize,
                 int maxPointsPerLine = 32)
      : LocalMapper(gridHeight, gridWidth, resolution, laserscanPosition,
                    laserscanOrientation, maxPointsPerLine),
        m_scanSize(scanSize) {
    m_q = sycl::queue{sycl::default_selector_v,
                      sycl::property::queue::in_order{}};
    auto dev = m_q.get_device();
    LOG_INFO("Running on :", dev.get_info<sycl::info::device::name>());
    m_devicePtrRanges = sycl::malloc_device<double>(scanSize, m_q);
    m_devicePtrAngles = sycl::malloc_device<double>(scanSize, m_q);
    m_devicePtrGrid = sycl::malloc_device<int>(m_gridHeight * m_gridWidth, m_q);
    m_devicePtrDistances =
        sycl::malloc_shared<float>(m_gridHeight * m_gridWidth, m_q);

    // initialize distances
    Eigen::Vector3f destPointLocal;
    std::cout << "Resolution: " << resolution << std::endl;
    for (size_t i = 0; i < m_gridHeight; ++i) {
      for (size_t j = 0; j < m_gridWidth; ++j) {
        destPointLocal = gridToLocal({i, j});
        m_devicePtrDistances[i + j * m_gridWidth] =
            (destPointLocal - m_laserscanPosition).norm();
      }
    }
  }

  // Destructor
  ~LocalMapperGPU() {
    m_q.wait(); // wait for the queue to finish before freeing memory
    if (m_devicePtrGrid) {
      sycl::free(m_devicePtrGrid, m_q);
    }
    if (m_devicePtrRanges) {
      sycl::free(m_devicePtrRanges, m_q);
    }
    if (m_devicePtrAngles) {
      sycl::free(m_devicePtrAngles, m_q);
    }
    if (m_devicePtrDistances) {
      sycl::free(m_devicePtrDistances, m_q);
    }
  }

  /**
   * Use the GPU to processes Laserscan data (angles and ranges) to project on
   * a 2D grid using Bresenham line drawing for each Laserscan beam
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @param gridData      Current grid data
   */
  Eigen::MatrixXi &scanToGrid(const std::vector<double> &angles,
                              const std::vector<double> &ranges);

private:
  const int m_scanSize;
  double *m_devicePtrRanges;
  double *m_devicePtrAngles;
  int *m_devicePtrGrid;
  float *m_devicePtrDistances;
  sycl::queue m_q;
};
} // namespace Mapping
} // namespace Kompass
