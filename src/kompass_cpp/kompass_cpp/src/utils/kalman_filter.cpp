#define EIGEN_DONT_PARALLELIZE // TODO: resolve error when enabling Eigen
                               // parallelization
#include "utils/kalman_filter.h"
#include "utils/logger.h"

namespace Kompass {

LinearSSKalmanFilter::LinearSSKalmanFilter(const size_t num_states,
                                           const size_t num_inputs) {
  state.resize(num_states, 1);
  // State update eq: X_dot = A X + B U + Q
  A.resize(num_states, num_states);
  B.resize(num_states, num_inputs);
  Q.resize(num_states, num_states);

  // Observation eq: Z = H X + R
  H.resize(num_states, num_states);
  R.resize(num_states, num_states);

  P = Eigen::MatrixXf::Identity(num_states, num_states);
}

bool LinearSSKalmanFilter::setup(const Eigen::MatrixXf &A,
                                 const Eigen::MatrixXf &B,
                                 const Eigen::MatrixXf &Q,
                                 const Eigen::MatrixXf &H,
                                 const Eigen::MatrixXf &R) {

  if ((A.size() != this->A.size()) || (B.size() != this->B.size()) ||
      (Q.size() != this->Q.size()) || (H.size() != this->H.size()) ||
      (R.size() != this->R.size())) {
    LOG_ERROR("Cannot setup the KalmanFilter. Matrix size error. Expected the "
              "following sized: A=",
              this->A.size(), ", B=", this->B.size(), ", H=", this->H.size(),
              ", Q=", this->Q.size(), ", R=", this->R.size());
    return false;
  }
  this->A = A;
  this->B = B;
  this->H = H;
  this->R = R;
  this->Q = Q;
  this->system_initialized = true;
  return true;
}

void LinearSSKalmanFilter::setInitialState(
    const Eigen::VectorXf &initial_state) {
  if (initial_state.size() != this->state.size()) {
    LOG_ERROR("Cannot set initial state. Expected the "
              "following sized: ",
              this->state.size());
    throw std::length_error("Error Setting Initial State");
  }
  this->state = initial_state;
  this->state_initialized = true;
}

void LinearSSKalmanFilter::setA(const Eigen::MatrixXf &A) { this->A = A; }

void LinearSSKalmanFilter::estimate(const Eigen::MatrixXf &measurements,
                                    const Eigen::MatrixXf &inputs,
                                    const int numberSteps) {
  Eigen::MatrixXf A_transpose = this->A.transpose();
  Eigen::MatrixXf B_inputs = this->B * inputs;
  Eigen::MatrixXf predicted_state = this->state;
  // predict a new state after number of steps
  for (int i = 0; i < numberSteps; i++) {
    predicted_state = this->A * predicted_state + B_inputs;
    // Covariance Extrapolation Equation
    this->P = this->A * this->P * A_transpose + this->Q;
  }

  // Innovation Matrix
  Eigen::MatrixXf S = this->R + this->H * this->P * this->H.transpose();

  // Kalman Gain Matrix
  Eigen::MatrixXf K = this->P * this->H.transpose() * S.inverse();

  // Update the state
  this->state =
      predicted_state + K * (measurements - this->H * predicted_state);

  // Update the estimation uncertainty
  this->P = (Eigen::MatrixXf::Identity(this->P.rows(), this->P.cols()) -
             K * this->H) *
            this->P;
}

void LinearSSKalmanFilter::estimate(const Eigen::MatrixXf &measurements,
                                    const int numberSteps) {
  // estimate with zero inputs
  auto size = this->B.cols();
  Eigen::MatrixXf inputs;
  inputs.resize(size, 1);
  inputs = Eigen::MatrixXf::Zero(size, 1);
  this->estimate(measurements, inputs);
}

double LinearSSKalmanFilter::getState(const size_t state_index) {
  if (this->state_initialized) {
    return this->state(state_index, 0);
  } else {
    return 0;
  }
}

std::optional<Eigen::MatrixXf> LinearSSKalmanFilter::getState() {
  if (this->state_initialized and this->system_initialized) {
    return this->state;
  }
  return std::nullopt;
}
} // namespace Kompass
