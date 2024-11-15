#include "mapping/line_drawing.h"
#include "mapping/local_mapper.h"

#include <vector>


namespace Kompass {
namespace Mapping {
// Function to convert a point from local coordinates frame of the grid to grid
// indices
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

/**
 * @brief Fill an area around a point on the grid with given padding.
 *
 * @param gridData Grid to be filled (2D Eigen matrix).
 * @param gridPoint Grid point indices (i,j) as a Vector2i.
 * @param gridPadding Padding to be filled (number of cells).
 * @param indicator Value to be assigned to filled cells.
 */
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
/**
 * Processes Laserscan data (angles and ranges) to project on a 2D grid using
 * Bresenham line drawing for each Laserscan beam
 *
 * @param angles        LaserScan angles in radians
 * @param ranges         LaserScan ranges in meters
 * @param gridData      Current grid data
 * @param gridDataProb Current probabilistic grid data
 * @param centralPoint  Coordinates of the central point of the grid
 * @param resolution     Grid resolution
 * @param laserscanPosition Position of the LaserScan sensor w.r.t the robot
 * @param previousGridDataProb Previous value of the probabilistic grid data
 * @param pPrior         LaserScan model's prior probability value
 * @param pEmpty         LaserScan model's probability value of empty cell
 * @param pOccupied       LaserScan model's probability value of occupied cell
 * @param rangeSure        LaserScan model's certain range (m)
 * @param rangeMax         LaserScan model's max range (m)
 * @param wallSize          LaserScan model's padding size when hitting an
 * @param oddLogPPrior    Log Odds of the LaserScan model's prior probability
 */
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
      localToGrid(Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)), centralPoint, resolution);

  for (int i = 0; i < angles.size(); ++i) {

    float x =
        laserscanPosition(0) + ranges(i) * cos(laserscanOrientation + angles(i));
    float y =
        laserscanPosition(1) + ranges(i) * sin(laserscanOrientation + angles(i));
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

      } else {
        rayStopped = false;
      }
      fillGridAroundPoint(gridData, lastGridPoint, 0, 100);
    }
  }
}
} // namespace Mapping
} // namespace Kompass
