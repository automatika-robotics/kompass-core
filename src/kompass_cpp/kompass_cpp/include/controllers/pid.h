#pragma once

#include "controller.h"
#include "datatypes/parameter.h"

namespace Kompass {
namespace Control {

class PID : public Controller {
public:
  // Constructor
  PID(double kp, double ki, double kd);
  PID();

  // Destructor
  virtual ~PID();

  void reset();

  // Set PID coefficients
  void setPIDCoefficients(double kp, double ki, double kd);

  // compute pid control based on target/current values
  double compute(double target, double current, double deltaTime);

  // Getters for PID coefficients
  double getKp() const { return config.getParameter<double>("Kp"); }
  double getKi() const { return config.getParameter<double>("Ki"); }
  double getKd() const { return config.getParameter<double>("Kd"); }

  std::array<double, 3> getPIDCoefficients();

protected:
  // Nested class for pid parameters
  class PIDParameters : public ControllerParameters {
  public:
    PIDParameters() {
      addParameter("Kp", Parameter(1.0, 0.0, 10.0)); // Proportional gain
      addParameter("Ki", Parameter(0.0, 0.0, 10.0)); // Integral gain
      addParameter("Kd", Parameter(0.0, 0.0, 10.0)); // Derivative gain
    }
  };

  PIDParameters config;

  // Previous errors for integral and derivative calculations
  double previousError{0};
  double integral{0};
  double derivative{0};

  // Control action methods
  void performLinearControl(double currentVelx, double currentVely,
                            double deltaTime);
  void performAngularControl(double currentOmega, double deltaTime);
};

} // namespace Control
} // namespace Kompass
