#include "mapping/local_mapper.h"
#include "mapping/line_drawing.h"

#include <vector>

namespace Kompass {
namespace Mapping {

Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral,
                            const Eigen::Vector2i &centralPoint,
                            float resolution) {

  // Initialize result vector with zeros
  Eigen::Vector2i gridPoint;

  // Calculate grid point by rounding coordinates in the local frame to nearest
  // cell boundaries
  gridPoint(0) =
      centralPoint(0) + static_cast<int>(poseTargetInCentral(0) / resolution);
  gridPoint(1) =
      centralPoint(1) + static_cast<int>(poseTargetInCentral(1) / resolution);

  return gridPoint;
}

void fillGridAroundPoint(Eigen::MatrixXi &gridData,
                         const Eigen::Vector2i &gridPoint, int gridPadding,
                         int indicator) {
  int iStart = gridPoint.x() - gridPadding;
  int iEnd = gridPoint.x() + gridPadding;
  int jStart = gridPoint.y() - gridPadding;
  int jEnd = gridPoint.y() + gridPadding;

  for (int i = iStart; i <= iEnd; ++i) {
    if (i >= 0 && i < gridData.rows()) {
      for (int j = jStart; j <= jEnd; ++j) {
        if (j >= 0 && j < gridData.cols()) {
          gridData(i, j) = indicator;
        }
      }
    }
  }

  // Ensure the central point is filled
  int i = gridPoint(0);
  int j = gridPoint(1);
  if (i >= 0 && i < gridData.rows() && j >= 0 && j < gridData.cols()) {
    gridData(i, j) = indicator;
  }
}

void laserscanToGrid(const Eigen::VectorXd &angles,
                     const Eigen::VectorXd &ranges, Eigen::MatrixXi &gridData,
                     Eigen::MatrixXi &gridDataProb,
                     const Eigen::Vector2i &centralPoint, float resolution,
                     const Eigen::Vector3f &laserscanPosition,
                     float laserscanOrientation,
                     const Eigen::MatrixXi &previousGridDataProb, float pPrior,
                     float pEmpty, float pOccupied, float rangeSure,
                     float rangeMax, float wallSize, float oddLogPPrior) {

  Eigen::Vector2i startPoint =
      localToGrid(Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)),
                  centralPoint, resolution);

  for (int i = 0; i < angles.size(); ++i) {

    float x = laserscanPosition(0) +
              ranges(i) * cos(laserscanOrientation + angles(i));
    float y = laserscanPosition(1) +
              ranges(i) * sin(laserscanOrientation + angles(i));
    Eigen::Vector2i toPoint =
        localToGrid(Eigen::Vector2f(x, y), centralPoint, resolution);
    std::vector<Eigen::Vector2i> points;

    bresenham(startPoint, toPoint, points);

    bool rayStopped = true;
    Eigen::Vector2i lastGridPoint = points[0];

    for (auto &pt : points) {

      if (pt(0) >= 0 && pt(0) < gridData.rows() && pt(1) >= 0 &&
          pt(1) < gridData.cols()) {

        // non-baysian update
        gridData(pt(0), pt(1)) = std::max(gridData(pt(0), pt(1)), 0);


        // save last point drawn
        lastGridPoint(0) = pt(0);
        lastGridPoint(1) = pt(1);

      } else {
        rayStopped = false;
      }
    }
    if (rayStopped) {
      /* Force non-bayesian map to set ending_point to occupied.
      this guarantee that when visualizing the map
      the obstacles detected by the laser scan can be visualized instead
      of having a map that visualize empty and unknown zones only.*/
      fillGridAroundPoint(gridData, lastGridPoint, 0, 100);
    }
  }
}
} // namespace Mapping
} // namespace Kompass
