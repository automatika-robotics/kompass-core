#pragma once

#include <cmath>
#include <math.h>
#include <stdexcept>
#include <vector>

namespace Path {

// Structure for Position
struct State {
  double x; // Speed on x-asix (m/s)
  double y;
  double yaw; // angular velocity (rad/s)
  double speed;

  State(double poseX = 0.0, double poseY = 0.0, double PoseYaw = 0.0,
        double speedValue = 0.0)
      : x(poseX), y(poseY), yaw(PoseYaw), speed(speedValue) {}
};

// Structure for a point in 2D space
struct Point {
  double x; // X coordinate
  double y; // Y coordinate

  Point(double xCoord = 0.0, double yCoord = 0.0) : x(xCoord), y(yCoord) {}
};

// Structure for Path Control parameters
struct Path {
  std::vector<Point> points;  // List of points defining the path
  std::vector<Path> segments; // List of path segments

  Path(const std::vector<Point> &points = {}) : points(points) {}

  size_t getMaxNumSegments() { return segments.size() - 1; }

  bool endReached(State currentState, double minDist) {
    Point endPoint = points.back();
    double dist = sqrt(pow(endPoint.x - currentState.x, 2) +
                       pow(endPoint.y - currentState.y, 2));
    return dist <= minDist;
  }

  Point getEnd() const { return points.back(); }

  Point getStart() const { return points.front(); }

  double getEndOrientation() const {
    const Point &p1 = points[points.size() - 2];
    const Point &p2 = points[points.size() - 1];

    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;

    // Compute the angle in radians
    double angle = std::atan2(dy, dx);

    return angle;
  }

  double getStartOrientation() const {
    const Point &p1 = points[0];
    const Point &p2 = points[1];

    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;

    // Compute the angle in radians
    double angle = std::atan2(dy, dx);

    return angle;
  }

  double getOrientation(const size_t index) const {
    Point p1;
    Point p2;
    if (index < points.size()) {
      p1 = points[index];
      p2 = points[index + 1];
    } else {
      p1 = points[index - 1];
      p2 = points[index];
    }

    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;

    // Compute the angle in radians
    double angle = std::atan2(dy, dx);

    return angle;
  }

  static double distance(const Point &p1, const Point &p2) {
    return std::sqrt((p2.x - p1.x) * (p2.x - p1.x) +
                     (p2.y - p1.y) * (p2.y - p1.y));
  }

  double minDist(const std::vector<Point> &others) const {
    double minDist{0.0};
    for (auto point : points) {
      minDist = std::numeric_limits<double>::max();
      for (auto other_point : others) {
        double dist = distance(point, other_point);
        if (dist < minDist) {
          minDist = dist;
        }
      }
    }
    return minDist;
  }

  // Function to compute the total path length
  double totalPathLength() const {
    double totalLength = 0.0;

    for (size_t i = 1; i < points.size(); ++i) {
      totalLength += distance(points[i - 1], points[i]);
    }

    return totalLength;
  }

  Point getPointAtLength(const double length) const {
    double totalLength = totalPathLength();
    if (length <= totalLength or points.size() > 2) {
      double accumLength = 0.0;
      double twoPointDist = distance(points[0], points[1]);
      for (size_t i = 1; i < points.size(); ++i) {
        accumLength += distance(points[i - 1], points[i]);
        if (std::abs(accumLength - totalLength) < twoPointDist) {
          return points[i - 1];
        }
      }
    }
    return getEnd();
  }

  size_t getNumberPointsInLength(double length) const {
    double totalLength = 0.0;

    for (size_t i = 1; i < points.size(); ++i) {
      totalLength += distance(points[i - 1], points[i]);
      if (totalLength >= length) {
        return i;
      }
    }
    return points.size();
  }

  // Segment the path by a given segment path length [m]
  void segment(double pathSegmentLength) {
    segments.clear();
    double totalLength = totalPathLength();
    if (pathSegmentLength >= totalLength) {
      segments.push_back(Path(points));
    } else {
      int numberOfPoints = getNumberPointsInLength(pathSegmentLength);
      segmentBySegmentNumber(numberOfPoints);
    }
  }

  // Segment using a number of segments
  void segmentBySegmentNumber(int numSegments) {
    segments.clear();
    if (numSegments <= 0 || points.empty()) {
      throw std::invalid_argument(
          "Invalid number of segments or empty points vector.");
    }

    int segmentSize = points.size() / numSegments;
    int remainder = points.size() % numSegments;

    auto it = points.begin();
    for (int i = 0; i < numSegments; ++i) {
      std::vector<Point> segment_points;
      for (int j = 0; j < segmentSize; ++j) {
        segment_points.push_back(*it++);
      }
      if (remainder > 0) {
        segment_points.push_back(*it++);
        --remainder;
      }
      segments.push_back(Path(segment_points));
    }
  }

  // Segment using a segment points number
  void segmentByPointsNumber(int segmentLength) {
    segments.clear();
    if (segmentLength <= 0 || points.empty()) {
      throw std::invalid_argument(
          "Invalid segment length or empty points vector.");
    }

    for (size_t i = 0; i < points.size(); i += segmentLength) {
      std::vector<Point> segment_points;
      for (size_t j = i; j < i + segmentLength && j < points.size(); ++j) {
        segment_points.push_back(points[j]);
      }

      segments.push_back(Path(segment_points));
    }
  }
};

struct PathPosition {
  size_t index{0};         // Index of the point in the segment
  size_t segment_index{0}; // Index of the segment in the Path
  double segment_length{-1.0};
  double parallel_distance{0.0};
  double normal_distance{0.0};
  State state;
};

} // namespace Path
