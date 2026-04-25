#pragma once

#include <cmath>

class Angle {
public:
  /**
   * @brief Normalize the angle to [0, 2*pi).
   *  NOTE: For symmetric "shortest-angular-distance" use normalizeToMinusPiPlusPi(a - b) instead, subtracting two values from this function can return (-2pi, 2pi) which is rarely what callers want.
   * @param angle
   * @return double
   */
  inline static double normalizeTo02Pi(double angle) {
    angle = fmod(angle, 2 * M_PI);
    if (angle < 0) {
      angle += 2 * M_PI;
    }
    return angle;
  }

  inline static double normalizeToMinusPiPlusPi(double angle) {
    // Normalize the angle to [-pi, pi]
    angle = fmod(angle + M_PI, 2 * M_PI);
    if (angle < 0) {
      angle += 2 * M_PI;
    }
    angle -= M_PI;
    return angle;
  }
};
