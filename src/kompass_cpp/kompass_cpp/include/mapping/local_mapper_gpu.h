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
  LocalMapperGPU(const int gridHeight, const int gridWidth,
                 const float resolution,
                 const Eigen::Vector3f &laserscanPosition,
                 const float laserscanOrientation, const bool isPointCloud,
                 const int scanSize, const float angleStep,
                 const float maxHeight, const float minHeight,
                 const float rangeMax, const int maxPointsPerLine = 32)
      : LocalMapper(gridHeight, gridWidth, resolution, laserscanPosition,
                    laserscanOrientation, isPointCloud, scanSize, angleStep, maxHeight,
                    minHeight, rangeMax, maxPointsPerLine) {
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

  /**
   * Uses a GPU to Projects 3D point cloud data onto a 2D grid using Bresenham
   * line drawing.
   *
   * @param data        Flattened point cloud data (int8), typically in XYZ
   * format.
   * @param point_step  Number of bytes between each point in the data array.
   * @param row_step    Number of bytes between each row in the data array.
   * @param height      Number of rows (height of the point cloud).
   * @param width       Number of columns (width of the point cloud).
   * @param x_offset    Offset (in bytes) to the x-coordinate within a point.
   * @param y_offset    Offset (in bytes) to the y-coordinate within a point.
   * @param z_offset    Offset (in bytes) to the z-coordinate within a point.
   * @return            A 2D occupancy grid as an Eigen::MatrixXi.
   */
  Eigen::MatrixXi &scanToGrid(const std::vector<int8_t> &data, int point_step,
                              int row_step, int height, int width,
                              float x_offset, float y_offset, float z_offset);

private:
  double *m_devicePtrRanges;
  double *m_devicePtrAngles;
  int *m_devicePtrGrid;
  float *m_devicePtrDistances;
  sycl::queue m_q;
};
} // namespace Mapping
} // namespace Kompass
