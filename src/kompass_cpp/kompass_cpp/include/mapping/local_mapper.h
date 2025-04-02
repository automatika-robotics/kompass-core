#pragma once

#include <Eigen/Dense>

namespace Kompass {
namespace Mapping {

// Occupancy types for grid
enum class OccupancyType { UNEXPLORED = -1, EMPTY = 0, OCCUPIED = 100 };

class LocalMapper {
public:
  // Constructor with basic parameters
  LocalMapper(const int gridHeight, const int gridWidth, float resolution,
              const Eigen::Vector3f &laserscanPosition,
              float laserscanOrientation, int maxPointsPerLine,
              int maxNumThreads = 1)
      : m_gridHeight(gridHeight), m_gridWidth(gridWidth),
        m_resolution(resolution), m_laserscanOrientation(laserscanOrientation),
        m_pPrior(0.5f), m_pEmpty(0.4f), m_pOccupied(0.6f), m_rangeSure(1.0f),
        m_rangeMax(5.0f), m_wallSize(0.2f),
        m_maxPointsPerLine(maxPointsPerLine),
        m_centralPoint(std::round(gridHeight / 2) - 1,
                       std::round(gridWidth / 2) - 1),
        m_laserscanPosition(laserscanPosition), m_maxNumThreads(maxNumThreads) {
    m_startPoint = localToGrid(
        Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)));
    gridData = Eigen::MatrixXi(gridHeight, gridWidth);
    gridDataProb = Eigen::MatrixXf(gridHeight, gridWidth);
    previousGridDataProb = Eigen::MatrixXf(gridHeight, gridWidth);
    // initialize previous grid data
    previousGridDataProb.fill(m_pPrior);
  }

  // Constructor with additional baysian parameters
  LocalMapper(const int gridHeight, const int gridWidth, float resolution,
              const Eigen::Vector3f &laserscanPosition,
              float laserscanOrientation, float pPrior, float pOccupied,
              float pEmpty, float rangeSure, float rangeMax, float wallSize,
              int maxPointsPerLine, int maxNumThreads = 1)
      : m_gridHeight(gridHeight), m_gridWidth(gridWidth),
        m_resolution(resolution), m_laserscanOrientation(laserscanOrientation),
        m_pPrior(pPrior), m_pEmpty(pEmpty), m_pOccupied(pOccupied),
        m_rangeSure(rangeSure), m_rangeMax(rangeMax), m_wallSize(wallSize),
        m_maxPointsPerLine(maxPointsPerLine),
        m_centralPoint(std::round(gridHeight / 2) - 1,
                       std::round(gridWidth / 2) - 1),
        m_laserscanPosition(laserscanPosition), m_maxNumThreads(maxNumThreads) {
    m_startPoint = localToGrid(
        Eigen::Vector2f(laserscanPosition(0), laserscanPosition(1)));
    gridData = Eigen::MatrixXi(gridHeight, gridWidth);
    gridDataProb = Eigen::MatrixXf(gridHeight, gridWidth);
    previousGridDataProb = Eigen::MatrixXf(gridHeight, gridWidth);
    // initialize previous grid data
    previousGridDataProb.fill(m_pPrior);
  }

  // Default destructor
  virtual ~LocalMapper() = default;

  /**
   * @brief Transform a grid to be centered in egocentric view of the current
   * position given its previous position.
   *
   * @param current_position_in_previous_pose Current egocentric position for
   * the transformation.
   * @param current_yaw_orientation_in_previous_pose Current egocentric
   * orientation for the transformation.
   * @param previous_grid_data Previous grid data (pre-transformation).
   * @param unknown_value Value of unknown occupancy (prior value for grid
   * cells).
   *
   * @return Transformed grid.
   */
  void getPreviousGridInCurrentPose(
      const Eigen::Vector2f &currentPositionInPreviousPose,
      double currentOrientationInPreviousPose);

  /**
   * @brief Updates a grid cell occupancy probability using the LaserScanModel
   *
   * @param distance Hit point distance from the sensor (m)
   * @param currentRange Scan ray hit range (m)
   * @param previousProb Previous probability assigned to grid cell
   */
  float updateGridCellProbability(float distance, float currentRange,
                                  float previousProb);

  /**
   * Processes Laserscan data (angles and ranges) to project on a 2D grid
   * using Bresenham line drawing for each Laserscan beam
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @returns gridData      Current grid data
   */
  Eigen::MatrixXi &scanToGrid(const std::vector<double> &angles,
                              const std::vector<double> &ranges);

  /**
   * Processes Laserscan data (angles and ranges) to project on a 2D grid
   * using Bresenham line drawing for each Laserscan beam and baysian map
   * updates
   *
   * @param angles        LaserScan angles in radians
   * @param ranges         LaserScan ranges in meters
   * @returns gridDataProb Current probabilistic grid data
   */
  std::tuple<Eigen::MatrixXi &, Eigen::MatrixXf &>
  scanToGridBaysian(const std::vector<double> &angles,
                    const std::vector<double> &ranges);

protected:
  // Transforms a point from grid coordinate (i,j) to the local coordinates
  // frame of the grid (around the central cell) (x,y,z)
  Eigen::Vector3f gridToLocal(const Eigen::Vector2i &pointTargetInGrid,
                              float height = 0.0) {
    Eigen::Vector3f poseB;
    poseB(0) = (m_centralPoint(0) - pointTargetInGrid(0)) * m_resolution;
    poseB(1) = (m_centralPoint(1) - pointTargetInGrid(1)) * m_resolution;
    poseB(2) = height;

    return poseB;
  }

  // Function to convert a point from local coordinates frame of the grid to
  // grid indices
  Eigen::Vector2i localToGrid(const Eigen::Vector2f &poseTargetInCentral) {

    Eigen::Vector2i gridPoint;

    // Calculate grid point by rounding coordinates in the local frame to
    // nearest cell boundaries
    gridPoint(0) = m_centralPoint(0) +
                   static_cast<int>(poseTargetInCentral(0) / m_resolution);
    gridPoint(1) = m_centralPoint(1) +
                   static_cast<int>(poseTargetInCentral(1) / m_resolution);

    return gridPoint;
  }

  // Update method for scanToGrid
  void updateGrid_(const float angle, const float range);

  // Update method for scanToGridBaysian
  void updateGridBaysian_(const float angle, const float range);

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

protected:
  const int m_gridHeight;
  const int m_gridWidth;
  const float m_resolution;
  const float m_laserscanOrientation;
  const float m_pPrior;
  const float m_pEmpty;
  const float m_pOccupied;
  const float m_rangeSure;
  const float m_rangeMax;
  const float m_wallSize;
  const int m_maxPointsPerLine;
  const Eigen::Vector2i m_centralPoint;
  const Eigen::Vector3f m_laserscanPosition;
  Eigen::Vector2i m_startPoint;
  Eigen::MatrixXi gridData;
  Eigen::MatrixXf gridDataProb;
  Eigen::MatrixXf previousGridDataProb;

private:
  const int m_maxNumThreads;
};

} // namespace Mapping
} // namespace Kompass
