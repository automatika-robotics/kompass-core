#pragma once

#include <cmath>

class Angle {
public:
  static double normalizeTo0Pi(double angle) ;

  static double normalizeToMinusPiPlusPi(double angle);
};
