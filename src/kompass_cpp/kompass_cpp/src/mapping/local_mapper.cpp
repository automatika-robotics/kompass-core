#include "mapping/local_mapper.h"
#include "mapping/line_drawing.h"
#include "utils/threadpool.h"

#include <Eigen/SparseCore>
#include <mutex>
#include <vector>

namespace Kompass {
namespace Mapping {

// Mutex for grid update
static std::mutex s_gridMutex;
Eigen::Vector3f gridToLocal(const Eigen::Vector2i &pointTargetInGrid,
                            const Eigen::Vector2i &centralPoint,
                            double resolution, double height) {
  Eigen::Vector3f poseB;
  poseB(0) = centralPoint(0) - pointTargetInGrid(0) * resolution;
  poseB(1) = centralPoint(1) - pointTargetInGrid(1) * resolution;
  poseB(2) = height;

  return poseB;
}

Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral,
                            const Eigen::Vector2i &centralPoint,
                            float resolution) {

  Eigen::Vector2i gridPoint;

  // Calculate grid point by rounding coordinates in the local frame to nearest
  // cell boundaries
  gridPoint(0) =
      centralPoint(0) + static_cast<int>(poseTargetInCentral(0) / resolution);
  gridPoint(1) =
      centralPoint(1) + static_cast<int>(poseTargetInCentral(1) / resolution);

  return gridPoint;
}

Eigen::MatrixXf getPreviousGridInCurrentPose(
    const Eigen::Vector2f &current_position_in_previous_pose,
    double current_orientation_in_previous_pose,
    const Eigen::MatrixXf &previous_grid_data,
    const Eigen::Vector2i &central_point, int grid_width, int grid_height,
    float resolution, float unknown_value) {
  // The new center on the previous map
  Eigen::Vector2f current_center =
      localToGrid(current_position_in_previous_pose, central_point, resolution);

  // Getting the angle from the difference in quaternion vector
  double current_orientation_angle =
      -1 *
      current_orientation_in_previous_pose; // Negative for clockwise rotation

  // Create transformation matrix to translate and rotate the center of the grid
  Eigen::Matrix3f transformation_matrix;
  double cos_theta = cos(current_orientation_angle);
  double sin_theta = sin(current_orientation_angle);

  transformation_matrix << cos_theta, -sin_theta,
      0.5 * grid_height - current_center(1) +
          (current_center(0) * sin_theta - current_center(1) * cos_theta),
      sin_theta, cos_theta,
      0.5 * grid_width - current_center(0) -
          (current_center(0) * cos_theta + current_center(1) * sin_theta),
      0, 0, 1;

  // Initialize the result matrix with unknown_value
  Eigen::MatrixXf transformed_grid(grid_height, grid_width);
  transformed_grid.fill(unknown_value);

  for (int y = 0; y < grid_height; ++y) {
    for (int x = 0; x < grid_width; ++x) {
      // Compute the inverse transformation
      Eigen::Vector3d src_point(x, y, 1.0);
      Eigen::Vector3d dst_point = transformation_matrix.inverse() * src_point;

      // Bilinear interpolation coordinates
      double src_x = dst_point(0);
      double src_y = dst_point(1);

      if (src_x >= 0 && src_x < previous_grid_data.cols() - 1 && src_y >= 0 &&
          src_y < previous_grid_data.rows() - 1) {

        int x0 = static_cast<int>(floor(src_x));
        int y0 = static_cast<int>(floor(src_y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float w0 = src_x - x0;
        float w1 = 1.0f - w0;
        float h0 = src_y - y0;
        float h1 = 1.0f - h0;

        float value = h1 * (w1 * previous_grid_data(y0, x0) +
                            w0 * previous_grid_data(y0, x1)) +
                      h0 * (w1 * previous_grid_data(y1, x0) +
                            w0 * previous_grid_data(y1, x1));

        transformed_grid(y, x) = value;
      }
    }
  }

  return transformed_grid;
}

void fillGridAroundPoint(Eigen::Ref<Eigen::MatrixXi> gridData,
                         const Eigen::Vector2i &gridPoint, int gridPadding,
                         int indicator) {
  // Calculate the valid range for i and j
  int rows = gridData.rows();
  int cols = gridData.cols();

  int iStart = std::max(0, gridPoint.x() - gridPadding);
  int iEnd = std::min(rows - 1, gridPoint.x() + gridPadding);
  int jStart = std::max(0, gridPoint.y() - gridPadding);
  int jEnd = std::min(cols - 1, gridPoint.y() + gridPadding);

  // Fill the grid within the calculated bounds
  for (int i = iStart; i <= iEnd; ++i) {
    for (int j = jStart; j <= jEnd; ++j) {
      gridData(i, j) = indicator;
    }
  }

  // Ensure the central point is filled (though it should be within bounds
  // already)
  int iCentral = gridPoint.x();
  int jCentral = gridPoint.y();
  if (iCentral >= 0 && iCentral < rows && jCentral >= 0 && jCentral < cols) {
    gridData(iCentral, jCentral) = indicator;
  }
}

float updateGridCellProbability(float distance, float currentRange,
                                float oddLogPPrev, float resolution,
                                float pPrior, float pEmpty, float pOccupied,
                                float rangeSure, float rangeMax, float wallSize,
                                float oddLogPPrior) {
  // get the current sensor probability of being occupied for an area in a given
  // distance from the scanner
  distance = distance * resolution;
  currentRange = currentRange - wallSize;

  float pF = (distance < currentRange) ? pEmpty : pOccupied;
  float delta = (distance < rangeSure) ? 0.0 : 1.0;

  float pSensor =
      pF + (delta * ((distance - rangeSure) / rangeMax) * (pPrior - pF));

  // get the current bayesian updated probability given its previous probability
  // and sensor probability
  float pCurr = oddLogPPrev + log(pSensor / (1.0 - pSensor)) - oddLogPPrior;

  return pCurr;
}

void updateGrid(const float angle, const float range,
                const Eigen::Vector2i &startPoint,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                Eigen::Ref<Eigen::MatrixXf> gridDataProb,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation,
                const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb,
                float pPrior, float pEmpty, float pOccupied, float rangeSure,
                float rangeMax, float wallSize, float oddLogPPrior,
                int maxPointsPerLine) {

  float x = laserscanPosition(0) + (range * cos(laserscanOrientation + angle));
  float y = laserscanPosition(1) + (range * sin(laserscanOrientation + angle));

  Eigen::Vector2i toPoint =
      localToGrid(Eigen::Vector2f(x, y), centralPoint, resolution);
  std::vector<Eigen::Vector2i> points;
  points.reserve(maxPointsPerLine);

  bresenhamEnhanced(startPoint, toPoint, points);

  bool rayStopped = true;
  Eigen::Vector2i lastGridPoint = points[0];

  // Precompute boundary checks
  int rows = gridData.rows();
  int cols = gridData.cols();
  int prevRows = previousGridDataProb.rows();
  int prevCols = previousGridDataProb.cols();

  // NOTE: when previous grid is transformed to align with the current
  // grid it's not perfectly aligned at the nearest grid cell.
  // Experimentally (empirically), a shift of 1 grid cell reverses the
  // mis-alignment effect. -> check test_local_grid_mapper.py for more
  // info
  int constexpr SHIFT = 1;

  for (auto &pt : points) {

    if (pt(0) >= 0 && pt(0) < rows && pt(1) >= 0 && pt(1) < cols) {

      // calculations for baysian update
      int x_prev = pt(0) + SHIFT;
      int y_prev = pt(1) + SHIFT;

      int previousValue =
          (x_prev >= 0 && x_prev < prevRows && y_prev >= 0 && y_prev < prevCols)
              ? previousGridDataProb(x_prev, y_prev)
              : oddLogPPrior;

      float distance = (pt - startPoint).norm();

      float newValue = updateGridCellProbability(
          distance, range, previousValue, resolution, pPrior, pEmpty, pOccupied,
          rangeSure, rangeMax, wallSize, oddLogPPrior);

      // grid updates
      {
        std::lock_guard<std::mutex> lock(s_gridMutex);
        // non-baysian update
        gridData(pt(0), pt(1)) = std::max(
            gridData(pt(0), pt(1)), static_cast<int>(OccupancyType::EMPTY));
        // baysian update
        gridDataProb(pt(0), pt(1)) =
            std::max(gridDataProb(pt(0), pt(1)), newValue);
      }

      // update last point drawn
      lastGridPoint(0) = pt(0);
      lastGridPoint(1) = pt(1);
    } else {
      rayStopped = false;
    }
  }
  if (rayStopped) {
    std::lock_guard<std::mutex> lock(s_gridMutex);
    /* Force non-bayesian map to set ending_point to occupied. So that
       when visualizing the map the detected obstacles can be visualized
       instead of having a map that visualize empty and unknown zones only.*/
    fillGridAroundPoint(gridData, lastGridPoint, 0,
                        static_cast<int>(OccupancyType::OCCUPIED));
  }
}

void scanToGrid(const std::vector<double> &angles,
                const std::vector<double> &ranges,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                Eigen::Ref<Eigen::MatrixXf> gridDataProb,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation,
                const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb,
                float pPrior, float pEmpty, float pOccupied, float rangeSure,
                float rangeMax, float wallSize, float oddLogPPrior,
                int maxPointsPerLine, int maxNumThreads) {

  // angles and ranges are handled as floats implicitly
  Eigen::Vector2i startPoint =
      localToGrid(Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)),
                  centralPoint, resolution);

  if (maxNumThreads > 1) {
    ThreadPool pool(maxNumThreads);
    for (int i = 0; i < angles.size(); ++i) {
      pool.enqueue(&updateGrid, angles[i], ranges[i], startPoint,
                   std::ref(gridData), std::ref(gridDataProb), centralPoint,
                   resolution, laserscanPosition, laserscanOrientation,
                   std::ref(previousGridDataProb), pPrior, pEmpty, pOccupied,
                   rangeSure, rangeMax, wallSize, oddLogPPrior,
                   maxPointsPerLine);
    }
  } else {
    for (int i = 0; i < angles.size(); ++i) {
      updateGrid(angles[i], ranges[i], startPoint, gridData, gridDataProb,
                 centralPoint, resolution, laserscanPosition,
                 laserscanOrientation, previousGridDataProb, pPrior, pEmpty,
                 pOccupied, rangeSure, rangeMax, wallSize, oddLogPPrior,
                 maxPointsPerLine);
    }
  }
}
} // namespace Mapping
} // namespace Kompass
