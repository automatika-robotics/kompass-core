#include "datatypes/path.h"
#include "utils/logger.h"
#include <stdexcept>
#include <vector>

using namespace std;
using namespace tk;

namespace Path {

Path::Path(const std::vector<Point> &points) : points(points) {
  for (const auto &point : points) {
    X_.emplace_back(point.x());
    Y_.emplace_back(point.y());
    Z_.emplace_back(point.z());
  }
}

const std::vector<float>& Path::getX() const { return X_; }

const std::vector<float>& Path::getY() const { return Y_; }

const std::vector<float>& Path::getZ() const { return Z_; }

void Path::setMaxLength(double max_length) {
  this->_max_path_length = max_length;
}

bool Path::endReached(State currentState, double minDist) {
  Point endPoint = points.back();
  double dist = sqrt(pow(endPoint.x() - currentState.x, 2) +
                     pow(endPoint.y() - currentState.y, 2));
  return dist <= minDist;
}

size_t Path::getMaxNumSegments() { return segments.size() - 1; }

Point Path::getEnd() const { return points.back(); }

Point Path::getStart() const { return points.front(); }

float Path::getEndOrientation() const {
  const Point &p1 = points[points.size() - 2];
  const Point &p2 = points[points.size() - 1];

  float dx = p2.x() - p1.x();
  float dy = p2.y() - p1.y();

  // Compute the angle in radians
  float angle = atan2(dy, dx);

  return angle;
}

float Path::getStartOrientation() const {
  const Point &p1 = points[0];
  const Point &p2 = points[1];

  float dx = p2.x() - p1.x();
  float dy = p2.y() - p1.y();

  // Compute the angle in radians
  float angle = atan2(dy, dx);

  return angle;
}

float Path::getOrientation(const size_t index) const {
  Point p1;
  Point p2;
  if (index < points.size()) {
    p1 = points[index];
    p2 = points[index + 1];
  } else {
    p1 = points[index - 1];
    p2 = points[index];
  }

  float dx = p2.x() - p1.x();
  float dy = p2.y() - p1.y();

  // Compute the angle in radians
  float angle = atan2(dy, dx);

  return angle;
}

float Path::distance(const Point &p1, const Point &p2) {
  return sqrt(pow((p2.x() - p1.x()), 2) + pow((p2.y() - p1.y()), 2));
}

// Function to compute the total path length
float Path::totalPathLength() const {

  if (points.empty()) {
    return 0.0;
  }

  float totalLength = 0.0;
  for (size_t i = 1; i < points.size(); ++i) {
    totalLength += distance(points[i - 1], points[i]);
  }

  return totalLength;
}

Point Path::getPointAtLength(const double length) const {
  float totalLength = totalPathLength();
  if (length <= totalLength or points.size() > 2) {
    float accumLength = 0.0;
    float twoPointDist = distance(points[0], points[1]);
    for (size_t i = 1; i < points.size(); ++i) {
      accumLength += distance(points[i - 1], points[i]);
      if (abs(accumLength - totalLength) < twoPointDist) {
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

  this->_max_interpolation_dist = max_interpolation_point_dist;
  // Set the maximum size for the points
  this->max_size = static_cast<size_t>(this->_max_path_length /
                                       this->_max_interpolation_dist);
  if (points.size() < 2) {
    throw invalid_argument(
        "At least two points are required to perform interpolation.");
  }

  // Get copies of X and Y vectors
  std::vector<float> x = getX();
  std::vector<float> y = getY();

  points.clear();
  X_.clear();
  Y_.clear();
  Z_.clear();

  float dist, x_e, y_e;

  for (size_t i = 0; i < x.size() - 1; ++i) {
    // Add the first point
    points.emplace_back(x[i], y[i], 0.0);
    X_.emplace_back(x[i]);
    Y_.emplace_back(y[i]);
    Z_.emplace_back(0.0);
    std::vector<double> x_points, y_points;

    // Add mid point to send 3 points for spline interpolation
    float mid_x = (x[i + 1] + x[i]) / 2;
    float mid_y;

    // Spline interpolation requires sorted data (x points)
    if (x[i + 1] > x[i]) {
      mid_y = y[i] + (mid_x - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
      x_points = {x[i], mid_x, x[i + 1]};
      y_points = {y[i], mid_y, y[i + 1]};
    } else {
      mid_y =
          y[i + 1] + (mid_x - x[i + 1]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
      x_points = {x[i + 1], mid_x, x[i]};
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

    x_e = x[i];
    dist = distance({x[i], y[i], 0.0}, {x[i + 1], y[i + 1], 0.0});
    int j = 1;

    // Interpolate new points between i, i+1
    while (dist > max_interpolation_point_dist and
           j < max_interpolation_iterations) {
      x_e = x[i] + j * (x[i + 1] - x[i]) * max_interpolation_point_dist;

      y_e = _spline->operator()(x_e);
      points.emplace_back(x_e, y_e, 0.0);
      X_.emplace_back(x_e);
      Y_.emplace_back(y_e);
      Z_.emplace_back(0.0);
      dist = distance({x_e, y_e, 0.0}, {x[i + 1], y[i + 1], 0.0});
      j++;
    }
    if (points.size() > this->max_size) {
      float remaining_len =
          distance({x[i + 1], y[i + 1], 0.0}, {x[-1], y[-1], 0.0});
      LOG_WARNING("Cannot interpolate more than ", this->max_size,
                  " path points -> "
                  "Discarding all future points of length ",
                  remaining_len, "m");
      break;
    }
  }
  if (points.size() < this->max_size) {
    // Add last point
    points.emplace_back(x.back(), y.back(), 0.0);
    X_.emplace_back(x.back());
    Y_.emplace_back(y.back());
    Z_.emplace_back(0.0);
  }
}

// Segment the path by a given segment path length [m]
void Path::segment(double pathSegmentLength) {
  if (_max_interpolation_dist > 0.0) {
    this->max_segment_size =
        static_cast<int>(pathSegmentLength / _max_interpolation_dist) + 1;
  }
  segments.clear();
  double totalLength = totalPathLength();
  Path new_segment;
  if (pathSegmentLength >= totalLength) {
    new_segment = Path(points);
    segments.push_back(new_segment);
  } else {
    int segmentsNumber = max(totalLength / pathSegmentLength, 1.0);
    if (segmentsNumber == 1) {
      new_segment = Path(points);
      new_segment.max_size = this->max_segment_size;
      segments.push_back(new_segment);
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

  this->max_segment_size = points.size() / numSegments;
  int remainder = points.size() % numSegments;

  auto it = points.begin();
  Path new_segment;
  for (int i = 0; i < numSegments; ++i) {
    std::vector<Point> segment_points;
    for (int j = 0; j < this->max_segment_size; ++j) {
      segment_points.push_back(*it++);
    }
    if (remainder > 0) {
      segment_points.push_back(*it++);
      --remainder;
    }
    new_segment = Path(segment_points);
    new_segment.max_size = this->max_segment_size;
    segments.push_back(new_segment);
  }
}

// Segment using a segment points number
void Path::segmentByPointsNumber(int segmentLength) {
  this->max_segment_size = segmentLength;
  segments.clear();
  if (segmentLength <= 0 || points.empty()) {
    throw std::invalid_argument(
        "Invalid segment length or empty points vector.");
  }
  Path new_segment;
  for (size_t i = 0; i < points.size(); i += segmentLength) {
    std::vector<Point> segment_points;
    for (size_t j = i; j < i + segmentLength && j < points.size(); ++j) {
      segment_points.push_back(points[j]);
    }
    new_segment = Path(segment_points);
    new_segment.max_size = this->max_segment_size;
    segments.push_back(new_segment);
  }
}
} // namespace Path
