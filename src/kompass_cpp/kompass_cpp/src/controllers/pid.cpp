#include <array>
#include "controllers/pid.h"

namespace Kompass {
namespace Control {

PID::PID(double kpValue, double kiValue, double kdValue)
    : previousError(0.0), integral(0.0), derivative(0.0) {
  config.setParameter("Kp", kpValue);
  config.setParameter("Ki", kiValue);
  config.setParameter("Kd", kdValue);
}

PID::PID()
    : previousError(0.0), integral(0.0), derivative(0.0) {
  // Default parameters are already set in PIDParameters constructor
}

PID::~PID() {}

void PID::reset(){
  this->derivative = 0;
  this->integral = 0;
}

void PID::setPIDCoefficients(double kpValue, double kiValue,
                                       double kdValue) {
  config.setParameter<double>("Kp", kpValue);
  config.setParameter<double>("Ki", kiValue);
  config.setParameter<double>("Kd", kdValue);
}

std::array<double, 3> PID::getPIDCoefficients(){
  return {getKp(), getKi(), getKd()};
}

double PID::compute(double target, double current, double deltaTime) {
  double currError = target - current;

  this->derivative = currError - this->previousError;
  this->integral += currError;

  this->previousError = currError;
  double control = (this->getKp() * currError) +
                   (this->getKi() * this->integral * deltaTime) +
                   (this->getKd() * this->derivative / deltaTime);

  return control;
}

} // namespace Control
} // namespace Kompass
