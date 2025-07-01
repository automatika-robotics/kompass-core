#pragma once

#include <Eigen/Dense>
#include <optional>

using Eigen::MatrixXf;
using namespace std;

namespace Kompass {

class LinearSSKalmanFilter {
private:
  bool state_initialized = false, system_initialized = false;
  Eigen::MatrixXf state;
  Eigen::MatrixXf A; // state matrix
  Eigen::MatrixXf B; // input matrix
  Eigen::MatrixXf H; // observation matrix
  Eigen::MatrixXf P; // uncertainty
  Eigen::MatrixXf Q; // process noise
  Eigen::MatrixXf R; // observation noise

public:
  // constructor
  LinearSSKalmanFilter(const size_t num_states, const size_t num_inputs);

  // set up the filter
  bool setup(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B,
             const Eigen::MatrixXf &Q, const Eigen::MatrixXf &H,
             const Eigen::MatrixXf &R);

  void setInitialState(const Eigen::VectorXf &initial_state);

  // set A (sometimes sampling time will differ)
  void setA(const Eigen::MatrixXf &A);

  // state estimate
  void estimate(const Eigen::MatrixXf &z, const Eigen::MatrixXf &u,
                const int numberSteps = 1);

  void estimate(const Eigen::MatrixXf &z, const int numberSteps = 1);

  // read output from the state
  double getState(const size_t state_index);

  std::optional<Eigen::MatrixXf> getState();
};

} // namespace Kompass
