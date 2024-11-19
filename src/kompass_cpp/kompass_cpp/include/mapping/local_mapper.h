#pragma once

#include <Eigen/Dense>

namespace Kompass {
namespace Mapping {

// Occupancy types for grid
enum class OccupancyType { UNEXPLORED = -1, EMPTY = 0, OCCUPIED = 100 };

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
void fillGridAroundPoint(Eigen::Ref<Eigen::MatrixXi> gridData,
                         const Eigen::Vector2i &gridPoint, int gridPadding,
                         int indicator);

/**
 * @brief Updates a grid cell occupancy probability using the LaserScanModel
 *
 * @param distance Hit point distance from the sensor (m)
 * @param currentRange Scan ray max range (m)
 * @param oddLogPPrev Log Odds of the previous probability of the grid cell
 * occupancy
 * @param resolution Grid resolution (meter/cell)
 * @param pPrior Prior probability of the model
 * @param pEmpty Empty probability of the model
 * @param pOccupied Occupied probability of the model
 * @param rangeSure Certainty range of the model (m)
 * @param rangeMax Max range of the sensor (m)
 * @param wallSize Padding size of the model (m)
 * @param oddLogPPrior Log Odds of the prior probability
 *
 * @return Current occupancy probability
 */
float updateGridCellProbability(float distance, float currentRange,
                                 float oddLogPPrev, float resolution,
                                 float pPrior, float pEmpty, float pOccupied,
                                 float rangeSure, float rangeMax,
                                 float wallSize, float oddLogPPrior);
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
                int maxNumThreads = 1);

} // namespace Mapping
} // namespace Kompass
