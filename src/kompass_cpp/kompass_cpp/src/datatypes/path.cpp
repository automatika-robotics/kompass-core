#include "datatypes/path.h"
#include "utils/logger.h"
#include "utils/spline.h"
#include <cmath>
#include <cstdlib>
#include <memory>
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
    Curvature_(i) = 0.0f;
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
  Curvature_.resize(max_size_);
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

Point Path::getEnd() const { return getIndex(current_size_ - 1); }

Point Path::getStart() const { return getIndex(0); }

Point Path::getIndex(const size_t index) const {
  assert(index < current_size_ && "Index out of range");
  return Point(X_(index), Y_(index), Z_(index));
}

Path::View Path::getSegment(size_t segment_index) const {
  if (segment_index >= segment_indices_.size()) {
    throw std::out_of_range(
        "Invalid segment index. Maximum number of segments is " +
        std::to_string(segment_indices_.size() - 1) +
        ", but requested segment index is " + std::to_string(segment_index));
  }

  size_t start_idx = segment_indices_[segment_index];
  size_t end_idx;

  if (segment_index + 1 < segment_indices_.size()) {
    end_idx = segment_indices_[segment_index + 1] - 1;
  } else {
    end_idx = current_size_ - 1;
  }

  return getPart(start_idx, end_idx);
}

Path::View Path::getPart(const size_t start, const size_t end) const {
  if (start >= current_size_ || end >= current_size_ || start > end) {
    throw std::out_of_range(
        "Invalid range for path part. Maximum path size is " +
        std::to_string(current_size_) + ", but requested part start= " +
        std::to_string(start) + ", and requested end= " + std::to_string(end));
  }

  size_t length = end - start + 1;

  return Path::View(*this, start, length);
}

void Path::pushPoint(const Point &point) {
  if (current_size_ >= max_size_) {
    throw std::out_of_range(
        "Path is full and reached maximum allowed size (max size = " +
        std::to_string(max_size_) + "). Cannot add more points.");
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

double Path::getCurvature(const size_t index) const {
  if (index >= current_size_)
    return 0.0;
  return Curvature_(index);
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

  // Set Curvature vector to zero
  Curvature_.setZero(maxSize);

  // Remaining iteration when interpolating the path (interpolation points
  // between each two path points)
  max_interpolation_iterations_ =
      static_cast<size_t>((maxSize - current_size_) / (current_size_));

  // Reset to zero in order to re-fill with interpolated points
  X_ = Eigen::VectorXf::Zero(max_size_);
  Y_ = Eigen::VectorXf::Zero(max_size_);
  Z_ = Eigen::VectorXf::Zero(max_size_);
  current_size_ = 0;

  float x_e, y_e, dist = 0.0f;

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
    std::unique_ptr<tk::spline> spline;
    if (type == InterpolationType::LINEAR) {
      spline =
          std::make_unique<tk::spline>(x_points, y_points, tk::spline::linear);
    } else if (type == InterpolationType::CUBIC_SPLINE) {
      spline =
          std::make_unique<tk::spline>(x_points, y_points, tk::spline::cspline);
    } else {
      spline = std::make_unique<tk::spline>(x_points, y_points,
                                            tk::spline::cspline_hermite);
    }

    x_e = x[i];
    dist = distance({x[i], y[i], 0.0}, {x[i + 1], y[i + 1], 0.0});
    int j = 1;

    // Interpolate new points between i, i+1
    while (dist > max_interpolation_point_dist and
           j < max_interpolation_iterations_) {
      x_e = x[i] + j * (x[i + 1] - x[i]) * max_interpolation_point_dist;

      y_e = spline->operator()(x_e);

      X_[current_size_] = x_e;
      Y_[current_size_] = y_e;

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
  // compute curvature
  float k = 0.0f, dx = 0.0f, dx_old = 0.0f, dy = 0.0f, dy_old = 0.0f,
        ddy = 0.0f, ddx = 0.0f;
  for (size_t idx = 1; idx < this->max_size_ - 1; ++idx) {
    dx = X_[idx] - X_[idx - 1];
    dy = Y_[idx] - Y_[idx - 1];
    // Previous velocity based on step from prev2 -> prev
    ddx = dx - dx_old;
    ddy = dy - dy_old;

    // Calculate Curvature
    // k = (dx * ddy - ddx * dy) / (dx^2 + dy^2)^1.5
    float denominator = std::pow(dx * dx + dy * dy, 1.5);
    if (denominator > 1e-3) {
      k = (dx * ddy - ddx * dy) / denominator;
    } else {
      k = 0.0f;
    }

    Curvature_[idx] = k;

    dx_old = dx;
    dy_old = dy;
  }
}

// Segment the path by a given segment path length [m]
void Path::segment(double pathSegmentLength) {
  if (current_size_ == 0)
    return;
  if (max_interpolation_dist_ > 0.0) {
    this->max_segment_size =
        static_cast<int>(pathSegmentLength / max_interpolation_dist_) + 1;
  }

  segment_indices_.clear();
  double totalLength = totalPathLength();

  // single point or single segment
  if (pathSegmentLength >= totalLength || current_size_ == 1) {
    segment_indices_.push_back(0);
    return;
  } else {
    size_t segmentsNumber =
        static_cast<size_t>(totalLength / pathSegmentLength);
    if (segmentsNumber > current_size_) {
      segmentsNumber = current_size_;
    }

    for (size_t i = 0; i < segmentsNumber; ++i) {
      segment_indices_.push_back((i * current_size_) / segmentsNumber);
    }
  }
  LOG_INFO("Got number of segments: ", segment_indices_.size());
}

Point Path::getSegmentStart(size_t segment_index) const {
  if (segment_index >= segment_indices_.size()) {
    throw std::out_of_range("Invalid segment index.");
  }

  size_t start_idx = segment_indices_[segment_index];
  return getIndex(start_idx);
}

Point Path::getSegmentEnd(size_t segment_index) const {
  if (segment_index >= segment_indices_.size()) {
    throw std::out_of_range("Invalid segment index.");
  }
  size_t end_idx;

  if (segment_index + 1 < segment_indices_.size()) {
    end_idx = segment_indices_[segment_index + 1] - 1;
  } else {
    end_idx = current_size_ - 1;
  }
  return getIndex(end_idx);
}

size_t Path::getSegmentSize(size_t segment_index) const {
  if (segment_index >= segment_indices_.size()) {
    throw std::out_of_range(
        "Invalid segment index. Maximum number of segments is " +
        std::to_string(segment_indices_.size() - 1) +
        ", but requested segment index is " + std::to_string(segment_index));
  }
  size_t start_idx = segment_indices_[segment_index];
  size_t end_idx;

  if (segment_index + 1 < segment_indices_.size()) {
    end_idx = segment_indices_[segment_index + 1] - 1;
  } else {
    end_idx = current_size_ - 1;
  }
  return end_idx - start_idx + 1;
}

size_t Path::getNumSegments() const { return segment_indices_.size(); }

size_t Path::getSegmentStartIndex(size_t segment_index) const {
  if (segment_index >= segment_indices_.size()) {
    throw std::out_of_range(
        "Invalid segment index. Maximum number of segments is " +
        std::to_string(segment_indices_.size() - 1) +
        ", but requested segment index is " + std::to_string(segment_index));
  }
  return segment_indices_[segment_index];
}

size_t Path::getSegmentEndIndex(size_t segment_index) const {
  if (segment_index >= segment_indices_.size()) {
    throw std::out_of_range(
        "Invalid segment index. Maximum number of segments is " +
        std::to_string(segment_indices_.size() - 1) +
        ", but requested segment index is " + std::to_string(segment_index));
  }
  size_t end_idx;

  if (segment_index + 1 < segment_indices_.size()) {
    end_idx = segment_indices_[segment_index + 1] - 1;
  } else {
    end_idx = current_size_ - 1;
  }
  return end_idx;
}

} // namespace Path
