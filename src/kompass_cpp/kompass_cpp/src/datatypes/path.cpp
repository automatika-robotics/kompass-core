#include "datatypes/path.h"
#include "utils/spline.h"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace tk;

namespace Path {

Path::Path(const std::vector<Point> &points) {
  if (points.size() < 2) {
    throw std::invalid_argument(
        "At least two points are required to create a path.");
  }
  current_size_ = points.size();
  resize(current_size_);
  for (size_t i = 0; i < points.size(); ++i) {
    X_(i) = points[i].x();
    Y_(i) = points[i].y();
    Z_(i) = points[i].z();
    Curvature_(i) = 0.0f;
  }
}

Path::Path(const Eigen::VectorXf &x_points, const Eigen::VectorXf &y_points,
           const Eigen::VectorXf &z_points) {
  if (x_points.size() != y_points.size() ||
      x_points.size() != z_points.size()) {
    throw std::invalid_argument("X, Y and Z vectors must have the same size.");
  }
  if (x_points.size() < 2) {
    throw std::invalid_argument(
        "At least two points are required to create a path.");
  }
  current_size_ = x_points.size();
  resize(current_size_);
  X_.head(current_size_) = x_points;
  Y_.head(current_size_) = y_points;
  Z_.head(current_size_) = z_points;
}

void Path::resize(const size_t max_size) {
  X_.resize(max_size);
  Y_.resize(max_size);
  Z_.resize(max_size);
  Curvature_.resize(max_size);
  interpolated_ = false;
}

bool Path::endReached(State currentState, double minDist) {
  Point endPoint = getEnd();
  double dist = sqrt(pow(endPoint.x() - currentState.x, 2) +
                     pow(endPoint.y() - currentState.y, 2));
  return dist <= minDist;
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
  resize(current_size_ + 1);
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

// Function to compute the total path length
float Path::totalPathLength() const {

  if (current_size_ < 2) {
    return 0.0;
  }

  // If path is already interpolated -> length was already calculated
  if (interpolated_) {
    return current_total_length_;
  }

  float totalLength = 0.0;
  for (size_t i = 1; i < current_size_; ++i) {
    totalLength += distance(getIndex(i - 1), getIndex(i));
  }

  return totalLength;
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

  // Push the first point
  s_vals.push_back(0.0);
  x_vals.push_back(X_[0]);
  y_vals.push_back(Y_[0]);

  // Reset the effective path length (meters)
  current_total_length_ = 0.0;

  for (size_t i = 1; i < current_size_; ++i) {
    // Calculate the segment distance
    double seg_dist = std::hypot(X_[i] - X_[i - 1], Y_[i] - Y_[i - 1]);
    current_total_length_ += seg_dist;

    // Push points
    s_vals.push_back(current_total_length_);
    x_vals.push_back(X_[i]);
    y_vals.push_back(Y_[i]);
  }

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
  // Calculate exact size needed
  size_t new_size = static_cast<size_t>(current_total_length_ /
                                        max_interpolation_point_dist) +
                    1;

  // Resize Eigen vectors (this resets data, but we saved it in x_vals/y_vals)
  resize(new_size);
  accumulated_path_length_.resize(new_size);

  // Reset Curvature
  Curvature_.setZero(new_size);
  Z_.setZero(new_size);

  // --- Interpolation Loop ---
  size_t idx = 0;
  // Iterate by distance arc distance 's'
  for (double s = 0.0; s <= current_total_length_ && idx < new_size;
       s += max_interpolation_point_dist) {
    accumulated_path_length_[idx] = s;
    X_[idx] = static_cast<float>(spline_x(s));
    Y_[idx] = static_cast<float>(spline_y(s));
    idx++;
  }

  // Add the last point at effective_len
  if (idx < new_size && idx > 0) {
    X_[idx] = static_cast<float>(spline_x(current_total_length_));
    Y_[idx] = static_cast<float>(spline_y(current_total_length_));
    idx++;
  }

  interpolated_ = true;

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

    float val = dx * dx + dy * dy;
    float denominator = val * std::sqrt(val);

    if (denominator > 1e-3f) {
      pK[i] = (dx_old * ddy - ddx * dy_old) / denominator;
      pK[i] = (dx * ddy - ddx * dy) / denominator;
    } else {
      pK[i] = 0.0f;
    }

    dx_old = dx;
    dy_old = dy;
  }
}

void Path::segment(double pathSegmentLength, size_t maxPointsPerSegment) {

  if (current_size_ < 2)
    return;

  segment_indices_.clear();
  segment_indices_.push_back(0);

  size_t segmentStartIdx = 0;
  float segmentStartLength = accumulated_path_length_[0];

  for (size_t i = 1; i < current_size_; ++i) {
    const size_t pointsInSegment = i - segmentStartIdx + 1;
    const float segmentLength =
        accumulated_path_length_[i] - segmentStartLength;

    const bool lengthExceeded =
        (pathSegmentLength > 0.0 && segmentLength >= pathSegmentLength);

    const bool pointsExceeded =
        (maxPointsPerSegment > 0 && pointsInSegment > maxPointsPerSegment);

    // Start a new segment at i
    if (lengthExceeded || pointsExceeded) {
      segment_indices_.push_back(i);

      segmentStartIdx = i;
      segmentStartLength = accumulated_path_length_[i];
    }
  }

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
