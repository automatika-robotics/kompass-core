#include <cmath>
#include "utils/angles.h"

double Angle::normalizeTo0Pi(double angle) {
  // Normalize the angle to [0, 2*pi]
  angle = fmod(angle, 2 * M_PI);
  if (angle < 0) {
    angle += 2 * M_PI;
  }
  // If angle is greater than pi, subtract from 2*pi to get the range [0, pi]
  if (angle > 2 * M_PI) {
    angle = 2 * M_PI - angle;
  }
  return angle;
}

double Angle::normalizeToMinusPiPlusPi(double angle) {
  // Normalize the angle to [-pi, pi]
  angle = fmod(angle + M_PI, 2 * M_PI);
  if (angle < 0) {
    angle += 2 * M_PI;
  }
  angle -= M_PI;
  return angle;
}
