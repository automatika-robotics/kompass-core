#pragma once

#include <Eigen/Dense>

namespace Kompass {
namespace Mapping {

// Occupancy types for grid
enum class OccupancyType { UNEXPLORED = -1, EMPTY = 0, OCCUPIED = 100 };

// Transforms a point from grid coordinate (i,j) to the local coordinates frame
// of the grid (around the central cell) (x,y,z)

Eigen::Vector3f gridToLocal(const Eigen::Vector2i &pointTargetInGrid,
                            const Eigen::Vector2i &centralPoint,
                            double resolution, double height = 0.0);

// Function to convert a point from local coordinates frame of the grid to grid
// indices
Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral,
                            const Eigen::Vector2i &centralPoint,
                            float resolution);

/**
 * @brief Transform a grid to be centered in egocentric view of the current
 * position given its previous position.
 *
 * @param current_position_in_previous_pose Current egocentric position for the
 * transformation.
 * @param current_yaw_orientation_in_previous_pose Current egocentric
 * orientation for the transformation.
 * @param previous_grid_data Previous grid data (pre-transformation).
 * @param central_point Coordinates of the central grid point.
 * @param grid_width Grid size (width).
 * @param grid_height Grid size (height).
 * @param resolution Grid resolution (meter/cell).
 * @param unknown_value Value of unknown occupancy (prior value for grid cells).
 *
 * @return Transformed grid.
 */
Eigen::MatrixXf getPreviousGridInCurrentPose(
    const Eigen::Vector2f &currentPositionInPreviousPose,
    double currentOrientationInPreviousPose,
    const Eigen::MatrixXf &previousGridData,
    const Eigen::Vector2i &centralPoint, int gridWidth, int gridHeight,
    float resolution, float unknownValue);

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
 * @param previousProb Previous probability of the grid cell
 * occupancy
 * @param resolution Grid resolution (meter/cell)
 * @param pPrior Prior probability of the model
 * @param pEmpty Empty probability of the model
 * @param pOccupied Occupied probability of the model
 * @param rangeSure Certainty range of the model (m)
 * @param rangeMax Max range of the sensor (m)
 * @param wallSize Padding size of the model (m)
 * @return Current occupancy probability
 */
float updateGridCellProbability(float distance, float currentRange,
                                float previousProb, float resolution,
                                float pPrior, float pEmpty, float pOccupied,
                                float rangeSure, float rangeMax,
                                float wallSize);
/**
 * Processes Laserscan data (angles and ranges) to project on a 2D grid using
 * Bresenham line drawing for each Laserscan beam
 *
 * @param angles        LaserScan angles in radians
 * @param ranges         LaserScan ranges in meters
 * @param gridData      Current grid data
 * @param centralPoint  Coordinates of the central point of the grid
 * @param resolution     Grid resolution
 * @param laserscanPosition Position of the LaserScan sensor w.r.t the robot
 * @param oddLogPPrior    Log Odds of the LaserScan model's prior probability
 */
void scanToGrid(const std::vector<double> &angles,
                const std::vector<double> &ranges,
                Eigen::Ref<Eigen::MatrixXi> gridData,
                const Eigen::Vector2i &centralPoint, float resolution,
                const Eigen::Vector3f &laserscanPosition,
                float laserscanOrientation, int maxPointsPerLine,
                int maxNumThreads);

/**
 * Processes Laserscan data (angles and ranges) to project on a 2D grid using
 * Bresenham line drawing for each Laserscan beam and baysian map updates
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
void scanToGridBaysian(
    const std::vector<double> &angles, const std::vector<double> &ranges,
    Eigen::Ref<Eigen::MatrixXi> gridData,
    Eigen::Ref<Eigen::MatrixXf> gridDataProb,
    const Eigen::Vector2i &centralPoint, float resolution,
    const Eigen::Vector3f &laserscanPosition, float laserscanOrientation,
    const Eigen::Ref<const Eigen::MatrixXf> previousGridDataProb, float pPrior,
    float pEmpty, float pOccupied, float rangeSure, float rangeMax,
    float wallSize, int maxPointsPerLine, int maxNumThreads = 1);

} // namespace Mapping
} // namespace Kompass
