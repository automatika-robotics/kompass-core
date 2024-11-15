#include "mapping/line_drawing.h"
#include "mapping/local_mapper.h"

#include <vector>


namespace Kompass {
namespace Mapping {
// Function to convert a point from local coordinates frame of the grid to grid
// indices
Eigen::Vector2i localToGrid(const Eigen::Vector3f &poseTargetInCentral,
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
 * Processes LaserScan data (angles and ranges) to project on a 2D grid using
 * Bresenham line drawing for each LaserScan beam
 *
 * @param angles        LaserScan angles in radians
 * @param ranges         LaserScan ranges in meters
 * @param grid_data      Current grid data
 * @param grid_data_prob Current probabilistic grid data
 * @param central_point  Coordinates of the central point of the grid
 * @param resolution     Grid resolution
 * @param laser_scan_pose Pose of the LaserScan sensor w.r.t the robot
 * @param previous_grid_data_prob Previous value of the probabilistic grid data
 * @param p_prior         LaserScan model's prior probability value
 * @param p_empty         LaserScan model's probability value of empty cell
 * @param p_occupied       LaserScan model's probability value of occupied cell
 * @param range_sure        LaserScan model's certain range (m)
 * @param range_max         LaserScan model's max range (m)
 * @param wall_size          LaserScan model's padding size when hitting an
 * obstacle (m)
 * @param odd_log_p_prior    Log Odds of the LaserScan model's prior probability
 * value
 *
 * @return List of occupied grid cells
 */
void laserscanToGrid(const Eigen::VectorXd &angles,
                     const Eigen::VectorXd &ranges, Eigen::MatrixXi &gridData,
                     Eigen::MatrixXi &gridDataProb,
                     const Eigen::Vector2i &centralPoint, float resolution,
                     const Eigen::Vector3f &laserScanPose,
                     float laserScanOrientation,
                     const Eigen::MatrixXi &previousGridDataProb, float pPrior,
                     float pEmpty, float pOccupied, float rangeSure,
                     float rangeMax, float wallSize, float oddLogPPrior) {

  Eigen::Vector2i startPoint =
      localToGrid(laserScanPose, centralPoint, resolution);

  for (int i = 0; i < angles.size(); ++i) {

    float x =
        laserScanPose(0) + ranges(i) * cos(laserScanOrientation + angles(i));
    float y =
        laserScanPose(1) + ranges(i) * sin(laserScanOrientation + angles(i));
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
