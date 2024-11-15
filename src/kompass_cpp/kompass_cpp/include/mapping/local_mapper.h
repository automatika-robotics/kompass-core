#pragma once

#include <Eigen/Dense>

namespace Kompass {
namespace Mapping {

// Function to convert a point from local coordinates frame of the grid to grid
// indices
Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral,
                            const Eigen::Vector2i &centralPoint,
                            float resolution);

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
                         int indicator);

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
                     float rangeMax, float wallSize, float oddLogPPrior);

} // namespace Mapping
} //namespace Kompass
