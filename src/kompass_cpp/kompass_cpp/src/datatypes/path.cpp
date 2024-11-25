#include "datatypes/path.h"
#include <stdexcept>
#include <vector>

using namespace std;
using namespace tk;

namespace Path{

Path::Path(const std::vector<Point> &points) : points(points){}

bool Path::endReached(State currentState, double minDist) {
Point endPoint = points.back();
double dist = sqrt(pow(endPoint.x - currentState.x, 2) +
                    pow(endPoint.y - currentState.y, 2));
return dist <= minDist;
}

size_t Path::getMaxNumSegments() { return segments.size() - 1; }

Point Path::getEnd() const { return points.back(); }

Point Path::getStart() const { return points.front(); }

double Path::getEndOrientation() const {
  const Point &p1 = points[points.size() - 2];
  const Point &p2 = points[points.size() - 1];

  double dx = p2.x - p1.x;
  double dy = p2.y - p1.y;

  // Compute the angle in radians
  double angle = std::atan2(dy, dx);

  return angle;
}

double Path::getStartOrientation() const {
  const Point &p1 = points[0];
  const Point &p2 = points[1];

  double dx = p2.x - p1.x;
  double dy = p2.y - p1.y;

  // Compute the angle in radians
  double angle = std::atan2(dy, dx);

  return angle;
}

double Path::getOrientation(const size_t index) const {
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

double Path::distance(const Point &p1, const Point &p2) {
  return std::sqrt((p2.x - p1.x) * (p2.x - p1.x) +
                   (p2.y - p1.y) * (p2.y - p1.y));
}

double Path::minDist(const std::vector<Point> &others) const {
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
double Path::totalPathLength() const {

  if (points.empty()) {
    return 0.0;
  }

  double totalLength = 0.0;
  for (size_t i = 1; i < points.size(); ++i) {
    totalLength += distance(points[i - 1], points[i]);
  }

  return totalLength;
}

Point Path::getPointAtLength(const double length) const {
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

size_t Path::getNumberPointsInLength(double length) const {
  double totalLength = 0.0;

  for (size_t i = 1; i < points.size(); ++i) {
    totalLength += distance(points[i - 1], points[i]);
    if (totalLength >= length) {
      return i;
    }
  }
  return points.size();
}

void Path::interpolate(double max_interpolation_point_dist,
                             InterpolationType type) {
  if (points.size() < 2) {
    throw invalid_argument(
        "At least two points are required to perform interpolation.");
  }

  vector<double> x(points.size()), y(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    x[i] = points[i].x;
    y[i] = points[i].y;
  }

  points.clear();

  for (size_t i = 0; i < x.size() - 1; ++i) {
    // Add the first point
    points.push_back({x[i], y[i]});
    std::vector<double> x_points, y_points;

    // Add mid point to send 3 points for spline interpolation
    double mid_x = (x[i + 1] + x[i]) / 2;
    double mid_y;

    // Spline interpolation requires sorted data (x points)
    if (x[i+1] > x[i]){
      mid_y =
          y[i] + (mid_x - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
      x_points = {x[i], mid_x, x[i + 1]};
      y_points = {y[i], mid_y, y[i + 1]};
    }
    else{
      mid_y =
          y[i + 1] + (mid_x - x[i + 1]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
      x_points = {x[i + 1], mid_x,  x[i]};
      y_points = {y[i + 1], mid_y, y[i]};
    }

    // Create the spline object and set the x and y values
    if (type == InterpolationType::LINEAR) {
      _spline = new tk::spline(x_points, y_points, tk::spline::linear);
    } else if (type == InterpolationType::CUBIC_SPLINE) {
      _spline = new tk::spline(x_points, y_points, tk::spline::cspline);
    } else {
      _spline = new tk::spline(x_points, y_points, tk::spline::cspline_hermite);
    }

    double point_e = x[i];
    double y_e;
    double dist = distance(x[i], x[i + 1]);
    int j = 1;

    // Interpolate new points between i, i+1
    while (dist > max_interpolation_point_dist and j < 500) {
      point_e = x[i] + j * (x[i + 1] - x[i]) * max_interpolation_point_dist;

      y_e = _spline->operator()(point_e);
      points.push_back({point_e, y_e});
      dist = distance(point_e, x[i + 1]);
      j++;
    }
  }
  // Add last point
  points.push_back({x.back(), y.back()});
}

  // Segment the path by a given segment path length [m]
void  Path::segment(double pathSegmentLength) {
  segments.clear();
  double totalLength = totalPathLength();
  if (pathSegmentLength >= totalLength) {
    segments.push_back(Path(points));
  } else {
    int segmentsNumber = std::max(totalLength / pathSegmentLength, 1.0);
    if (segmentsNumber == 1) {
      segments.push_back(Path(points));
      return;
    }
    segmentBySegmentNumber(segmentsNumber);
  }
}

  // Segment using a number of segments
void Path::segmentBySegmentNumber(int numSegments) {
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
void Path::segmentByPointsNumber(int segmentLength) {
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
}
