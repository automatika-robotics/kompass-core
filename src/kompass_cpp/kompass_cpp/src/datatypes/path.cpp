#include "datatypes/path.h"
#include "utils/logger.h"
#include "utils/spline.h"
#include <cmath>
#include <cstdlib>
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
    throw std::invalid_argument(
        "At least two points are required to perform interpolation.");
  }

  // --- Data Preparation & Parametrization ---
  // Extract current points and compute 's' (cumulative distance)
  std::vector<double> s_vals;
  std::vector<double> x_vals;
  std::vector<double> y_vals;

  // Reserve memory
  s_vals.reserve(current_size_);
  x_vals.reserve(current_size_);
  y_vals.reserve(current_size_);

  double current_dist = 0.0;

  // Push the first point
  s_vals.push_back(0.0);
  x_vals.push_back(X_[0]);
  y_vals.push_back(Y_[0]);

  for (size_t i = 1; i < current_size_; ++i) {
    // Calculate the segment distance
    double seg_dist = std::hypot(X_[i] - X_[i - 1], Y_[i] - Y_[i - 1]);
    current_dist += seg_dist;

    // Push points
    s_vals.push_back(current_dist);
    x_vals.push_back(X_[i]);
    y_vals.push_back(Y_[i]);
  }

  double total_input_length = current_dist;

  // --- Global Spline Construction ---
  tk::spline spline_x, spline_y;
  tk::spline::spline_type sp_type;

  switch (type) {
  case InterpolationType::LINEAR:
    sp_type = tk::spline::linear;
    break;
  case InterpolationType::HERMITE_SPLINE:
    sp_type = tk::spline::cspline_hermite;
    break;
  case InterpolationType::CUBIC_SPLINE:
  default:
    sp_type = tk::spline::cspline;
    break;
  }

  spline_x.set_points(s_vals, x_vals, sp_type);
  spline_y.set_points(s_vals, y_vals, sp_type);

  // --- Output Vector Resizing ---
  this->max_interpolation_dist_ = max_interpolation_point_dist;

  // Cap the length to maximum allowed length
  double effective_len =
      std::min(static_cast<double>(max_path_length_), total_input_length);

  // Calculate exact size needed
  size_t new_size =
      static_cast<size_t>(effective_len / max_interpolation_point_dist) + 1;

  // Resize Eigen vectors (this resets data, but we saved it in x_vals/y_vals)
  resize(new_size);

  // Reset Curvature
  Curvature_.setZero(max_size_);

  // --- Interpolation Loop ---
  size_t idx = 0;
  // Iterate by distance arc distance 's'
  for (double s = 0.0; s <= effective_len && idx < max_size_;
       s += max_interpolation_point_dist) {
    X_[idx] = static_cast<float>(spline_x(s));
    Y_[idx] = static_cast<float>(spline_y(s));
    idx++;
  }

  // Add the last point at effective_len
  if (idx < max_size_ && idx > 0) {
    X_[idx] = static_cast<float>(spline_x(effective_len));
    Y_[idx] = static_cast<float>(spline_y(effective_len));
    idx++;
  }

  current_size_ = idx;

  // --- Optimized Curvature Calculation ---
  float *pX = X_.data();
  float *pY = Y_.data();
  float *pK = Curvature_.data();

  // first derivatives
  float dx_old = pX[1] - pX[0];
  float dy_old = pY[1] - pY[0];

  for (size_t i = 1; i < current_size_ - 1; ++i) {
    float dx = pX[i + 1] - pX[i];
    float dy = pY[i + 1] - pY[i];

    float ddx = dx - dx_old;
    float ddy = dy - dy_old;

    // Optimization: avoid pow(..., 1.5)
    float val = dx * dx + dy * dy;
    float denominator = val * std::sqrt(val);

    if (denominator > 1e-3f) {
      pK[i] = (dx_old * ddy - ddx * dy_old) /
              denominator;
      pK[i] = (dx * ddy - ddx * dy) / denominator;
    } else {
      pK[i] = 0.0f;
    }

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
