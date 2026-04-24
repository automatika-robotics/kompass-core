#pragma once

#include "datatypes/path.h"
#include "datatypes/trajectory.h"
#include <cmath>
#include <limits>
#include <vector>

// Shared helpers for the PurePursuit and DWA controller tests. Defined inline
// so each test translation unit gets its own copy without ODR issues.

inline void applyControl(Path::State &robotState,
                         const Kompass::Control::Velocity2D control,
                         const double timeStep) {
  double dx = (control.vx() * std::cos(robotState.yaw) -
               control.vy() * std::sin(robotState.yaw)) *
              timeStep;
  double dy = (control.vx() * std::sin(robotState.yaw) +
               control.vy() * std::cos(robotState.yaw)) *
              timeStep;
  double dyaw = control.omega() * timeStep;
  robotState.x += dx;
  robotState.y += dy;
  robotState.yaw += dyaw;

  // Normalize yaw
  while (robotState.yaw > M_PI)
    robotState.yaw -= 2.0 * M_PI;
  while (robotState.yaw < -M_PI)
    robotState.yaw += 2.0 * M_PI;
}

// --- Path Generators ---
inline Path::Path createStraightPath() {
  std::vector<Path::Point> points;
  for (double x = 0.0; x <= 10.0; x += 0.5) {
    points.emplace_back(x, 0.0, 0.0);
  }
  return Path::Path(points);
}

inline Path::Path createUTurnPath() {
  std::vector<Path::Point> points;
  // First straight segment
  for (double x = 0.0; x <= 5.0; x += 0.5) {
    points.emplace_back(x, 0.0, 0.0);
  }
  // Semi-circle turn
  double radius = 5.5;
  double center_x = 5.0;
  double center_y = 2.5;
  for (double angle = -M_PI_2; angle <= M_PI_2; angle += 0.2) {
    points.emplace_back(center_x + radius * std::cos(angle),
                        center_y + radius * std::sin(angle), 0.0);
  }
  // Return straight segment
  for (double x = 5.0; x >= 0.0; x -= 0.5) {
    points.emplace_back(x, 5.0, 0.0);
  }
  return Path::Path(points);
}

inline Path::Path createCirclePath() {
  // 3/4 of a circle
  std::vector<Path::Point> points;
  double radius = 10.0;
  for (double angle = 0.0; angle <= 3.0 * M_PI / 2.0; angle += 0.1) {
    points.emplace_back(radius * std::cos(angle), radius * std::sin(angle),
                        0.0);
  }
  return Path::Path(points);
}

// Creates a circular obstacle as a point cloud
inline std::vector<Path::Point> createRoundObstacle(double x, double y,
                                                    double radius,
                                                    double resolution = 0.1) {
  std::vector<Path::Point> cloud;
  for (double r = 0; r <= radius; r += resolution) {
    for (double theta = 0; theta < 2 * M_PI;
         theta +=
         resolution /
         r) { // Scale theta step by radius to keep density somewhat constant
      cloud.emplace_back(x + r * std::cos(theta), y + r * std::sin(theta), 0.0);
    }
    // Handle center point r=0 separately to avoid division by zero if loop
    // condition allows
    if (r == 0)
      cloud.emplace_back(x, y, 0.0);
  }
  return cloud;
}

inline double minDistanceToCloud(double x, double y,
                                 const std::vector<Path::Point> &cloud) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto &p : cloud) {
    double d = std::hypot(p.x() - x, p.y() - y);
    if (d < best)
      best = d;
  }
  return best;
}
