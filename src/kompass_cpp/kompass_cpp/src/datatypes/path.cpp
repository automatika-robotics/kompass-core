#include "datatypes/path.h"
#include "utils/logger.h"
#include <stdexcept>
#include <vector>

using namespace std;
using namespace tk;

namespace Path {

Path::Path(const std::vector<Point> &points, const size_t new_max_size) {
  if (new_max_size < 2) {
    throw std::invalid_argument(
        "At least two points are required to create a path.");
  }
  if (points.size() > new_max_size) {
    throw std::invalid_argument(
        "Points size is larger than the allowed maximum size.");
  }
  resize(new_max_size);
  current_size_ = points.size();
  for (size_t i = 0; i < points.size(); ++i) {
    X_(i) = points[i].x();
    Y_(i) = points[i].y();
    Z_(i) = points[i].z();
  }
}

Path::Path(const Eigen::VectorXf &x_points, const Eigen::VectorXf &y_points,
           const Eigen::VectorXf &z_points, const size_t new_max_size) {
  if (x_points.size() != y_points.size() ||
      x_points.size() != z_points.size()) {
    throw std::invalid_argument("X, Y and Z vectors must have the same size.");
  }
  if (new_max_size < 2) {
    throw std::invalid_argument(
        "At least two points are required to create a path.");
  }
  if (x_points.size() > new_max_size) {
    throw std::invalid_argument(
        "Points size is larger than the allowed maximum size.");
  }
  resize(new_max_size);
  current_size_ = x_points.size();
  X_.head(current_size_) = x_points;
  Y_.head(current_size_) = y_points;
  Z_.head(current_size_) = z_points;
}

const Eigen::VectorXf Path::getX() const {
  return X_.segment(0, current_size_);
}

const Eigen::VectorXf Path::getY() const {
  return Y_.segment(0, current_size_);
}

const Eigen::VectorXf Path::getZ() const {
  return Z_.segment(0, current_size_);
}

size_t Path::getSize() const { return current_size_; }

void Path::setMaxLength(double max_length) { max_path_length_ = max_length; }

void Path::resize(const size_t new_max_size) {
  max_size_ = new_max_size;
  X_.resize(max_size_);
  Y_.resize(max_size_);
  Z_.resize(max_size_);
  if (current_size_ > max_size_) {
    current_size_ = max_size_;
  }
}

bool Path::endReached(State currentState, double minDist) {
  Point endPoint = getEnd();
  double dist = sqrt(pow(endPoint.x() - currentState.x, 2) +
                     pow(endPoint.y() - currentState.y, 2));
  return dist <= minDist;
}

size_t Path::getMaxSize() const { return max_size_; }

size_t Path::getMaxNumSegments() { return segments.size() - 1; }

Point Path::getEnd() const { return getIndex(current_size_ - 1); }

Point Path::getStart() const { return getIndex(0); }

Point Path::getIndex(const size_t index) const {
  assert(index < current_size_ && "Index out of range");
  return Point(X_(index), Y_(index), Z_(index));
}

Path Path::getPart(const size_t start, const size_t end,
                   const size_t max_part_size) const {
  if (start >= current_size_ || end >= current_size_ || start >= end) {
    throw std::out_of_range("Invalid range for path part.");
  }
  auto part_size = std::max(max_part_size, end - start + 1);
  Path part(X_.segment(start, end - start + 1),
            Y_.segment(start, end - start + 1),
            Z_.segment(start, end - start + 1), part_size);
  return part;
}

void Path::pushPoint(const Point &point) {
  if (current_size_ >= max_size_) {
    throw std::out_of_range("Path is full. Cannot add more points.");
  }
  X_(current_size_) = point.x();
  Y_(current_size_) = point.y();
  Z_(current_size_) = point.z();
  current_size_++;
}

float Path::getEndOrientation() const {
  const Point &p1 = getIndex(current_size_ - 2);
  const Point &p2 = getIndex(current_size_ - 1);

  float dx = p2.x() - p1.x();
  float dy = p2.y() - p1.y();

  // Compute the angle in radians
  float angle = atan2(dy, dx);

  return angle;
}

float Path::getStartOrientation() const {
  const Point &p1 = getIndex(0);
  const Point &p2 = getIndex(1);

  float dx = p2.x() - p1.x();
  float dy = p2.y() - p1.y();

  // Compute the angle in radians
  float angle = atan2(dy, dx);

  return angle;
}

float Path::getOrientation(const size_t index) const {
  Point p1;
  Point p2;
  if (index < current_size_) {
    p1 = getIndex(index);
    p2 = getIndex(index + 1);
  } else {
    p1 = getIndex(index - 1);
    p2 = getIndex(index);
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

  if (current_size_ < 2) {
    return 0.0;
  }

  float totalLength = 0.0;
  for (size_t i = 1; i < current_size_; ++i) {
    totalLength += distance(getIndex(i - 1), getIndex(i));
  }

  return totalLength;
}

Point Path::getPointAtLength(const double length) const {
  float totalLength = totalPathLength();
  if (length <= totalLength or current_size_ > 2) {
    float accumLength = 0.0;
    float twoPointDist = distance(getIndex(0), getIndex(1));
    for (size_t i = 1; i < current_size_; ++i) {
      accumLength += distance(getIndex(i - 1), getIndex(i));
      if (abs(accumLength - totalLength) < twoPointDist) {
        return getIndex(i - 1);
      }
    }
  }
  return getEnd();
}

size_t Path::getNumberPointsInLength(double length) const {
  double totalLength = 0.0;

  for (size_t i = 1; i < current_size_; ++i) {
    totalLength += distance(getIndex(i - 1), getIndex(i));
    if (totalLength >= length) {
      return i;
    }
  }
  return current_size_;
}

void Path::interpolate(double max_interpolation_point_dist,
                       InterpolationType type) {
  if (current_size_ < 2) {
    throw invalid_argument(
        "At least two points are required to perform interpolation.");
  }
  // Get copies of X and Y vectors effective points
  Eigen::VectorXf x = X_.segment(0, current_size_);
  Eigen::VectorXf y = Y_.segment(0, current_size_);

  this->max_interpolation_dist_ = max_interpolation_point_dist;
  // Set the maximum size for the points
  auto maxSize = static_cast<size_t>(this->max_path_length_ /
                                     this->max_interpolation_dist_);
  resize(maxSize);
  // Remaining iteration when interpolating the path (interpolation points
  // between each two path points)
  max_interpolation_iterations_ =
      static_cast<size_t>((maxSize - current_size_) / (current_size_));

  Z_ = Eigen::VectorXf::Zero(max_size_);
  current_size_ = 0;

  float dist, x_e, y_e;

  for (size_t i = 0; i < x.size() - 1; ++i) {
    // Add the first point
    X_(current_size_) = x[i];
    Y_(current_size_) = y[i];
    current_size_++;
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
      spline_ = new tk::spline(x_points, y_points, tk::spline::linear);
    } else if (type == InterpolationType::CUBIC_SPLINE) {
      spline_ = new tk::spline(x_points, y_points, tk::spline::cspline);
    } else {
      spline_ = new tk::spline(x_points, y_points, tk::spline::cspline_hermite);
    }

    x_e = x[i];
    dist = distance({x[i], y[i], 0.0}, {x[i + 1], y[i + 1], 0.0});
    int j = 1;

    // Interpolate new points between i, i+1
    while (dist > max_interpolation_point_dist and
           j < max_interpolation_iterations_) {
      x_e = x[i] + j * (x[i + 1] - x[i]) * max_interpolation_point_dist;

      y_e = spline_->operator()(x_e);
      X_(current_size_) = x_e;
      Y_(current_size_) = y_e;
      current_size_++;
      dist = distance({x_e, y_e, 0.0}, {x[i + 1], y[i + 1], 0.0});
      j++;
    }
    if (current_size_ > this->max_size_) {
      float remaining_len =
          distance({x[i + 1], y[i + 1], 0.0}, {x[-1], y[-1], 0.0});
      LOG_WARNING("Cannot interpolate more than ", this->max_size_,
                  " path points -> "
                  "Discarding all future points of length ",
                  remaining_len, "m");
      break;
    }
  }
  if (current_size_ < this->max_size_) {
    // Add last point
    X_(current_size_) = x(x.size() - 1);
    Y_(current_size_) = y(y.size() - 1);
    current_size_++;
  }
}

// Segment the path by a given segment path length [m]
void Path::segment(double pathSegmentLength) {
  if (max_interpolation_dist_ > 0.0) {
    this->max_segment_size =
        static_cast<int>(pathSegmentLength / max_interpolation_dist_) + 1;
  }
  segments.clear();
  double totalLength = totalPathLength();
  if (pathSegmentLength >= totalLength) {
    // Add the whole path as a single segment
    auto new_segment = *this;
    new_segment.resize(this->max_segment_size);
    segments.push_back(new_segment);
  } else {
    int segmentsNumber =
        max(static_cast<int>(totalLength / pathSegmentLength), 1);
    if (segmentsNumber == 1) {
      auto new_segment = *this;
      new_segment.resize(this->max_segment_size);
      segments.push_back(new_segment);
      return;
    }
    segmentBySegmentNumber(segmentsNumber);
  }
}

// Segment using a number of segments
void Path::segmentBySegmentNumber(int numSegments) {
  segments.clear();
  if (numSegments <= 0 || current_size_ <= 0) {
    throw std::invalid_argument(
        "Invalid number of segments or empty points vector.");
  }

  this->max_segment_size = current_size_ / numSegments;
  int remainder = current_size_ % numSegments;

  auto it_x = X_.begin();
  auto it_y = Y_.begin();
  auto it_z = Z_.begin();
  for (int i = 0; i < numSegments; ++i) {
    std::vector<Point> segment_points;
    for (int j = 0; j < this->max_segment_size; ++j) {
      segment_points.push_back({*it_x++, *it_y++, *it_z++});
    }
    if (remainder > 0) {
      segment_points.push_back({*it_x++, *it_y++, *it_z++});
      --remainder;
    }
    auto new_segment = Path(segment_points, segment_points.size());
    segments.push_back(new_segment);
  }
}

// Segment using a segment points number
void Path::segmentByPointsNumber(int segmentLength) {
  this->max_segment_size = segmentLength;
  segments.clear();
  if (segmentLength <= 0 || current_size_ <= 0) {
    throw std::invalid_argument(
        "Invalid segment length or empty points vector.");
  }
  for (size_t i = 0; i < current_size_; i += segmentLength) {
    std::vector<Point> segment_points;
    for (size_t j = i; j < i + segmentLength && j < current_size_; ++j) {
      segment_points.push_back(getIndex(j));
    }
    auto new_segment = Path(segment_points, segment_points.size());
    new_segment.max_size_ = this->max_segment_size;
    segments.push_back(new_segment);
  }
}
} // namespace Path
