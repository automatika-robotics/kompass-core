# pragma once
#include <chrono>
#include "utils/logger.h"
#include <vector>
#include <cmath>
#include "utils/angles.h"
#include <cstring> // For memcpy

#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif /* _COLORS_ */


using namespace Kompass;

struct Timer {
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point end;
  std::chrono::duration<float> duration;

  Timer() {
    start = std::chrono::high_resolution_clock::now();
    Logger::getInstance().setLogLevel(LogLevel::DEBUG);
  }
  ~Timer() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    LOG_INFO(BOLD(FBLU("Time taken for test: ")), KBLU,
             duration.count() * 1000.0f, RST, BOLD(FBLU(" ms\n")));
  }
};

inline void initLaserscan(size_t N, double initRange, std::vector<double> &ranges,
                          std::vector<double> &angles) {
  angles.resize(N);
  ranges.resize(N);
  for (size_t i = 0; i < N; ++i) {
    angles[i] = 2.0 * M_PI * static_cast<double>(i) / N;
    ranges[i] = initRange;
  }
}

inline size_t findClosestIndex_(double angle, std::vector<double> &angles) {
  // Normalize the angle to be within [0, 2*pi)
  angle = Angle::normalizeTo0Pi(angle);

  double minDiff = 2.0 * M_PI;
  size_t closestIndex = 0;

  for (size_t i = 0; i < angles.size(); ++i) {
    double diff = std::abs(angles[i] - angle);
    if (diff < minDiff) {
      minDiff = diff;
      closestIndex = i;
    }
  }

  return closestIndex;
}

inline void setLaserscanAtAngle(double angle, double rangeValue,
                                std::vector<double> &ranges,
                                std::vector<double> &angles) {
  // Find the closest index to the given angle
  size_t index = findClosestIndex_(angle, angles);

  if (index < ranges.size()) {
    ranges[index] = rangeValue;
  } else {
    LOG_ERROR("Angle Index is out of bounds, got  ", index, "and size is ",
              ranges.size());
  }
}

// Helper for creating Fake PointCloud Data
inline void addPointToCloud(std::vector<int8_t> &cloud_data, float x, float y, float z,
                     int point_step, int x_offset, int y_offset, int z_offset) {
  size_t current_size = cloud_data.size();
  cloud_data.resize(current_size + point_step, 0);

  int8_t *point_ptr = cloud_data.data() + current_size;

  // Copy floats to raw bytes
  std::memcpy(point_ptr + x_offset, &x, sizeof(float));
  std::memcpy(point_ptr + y_offset, &y, sizeof(float));
  std::memcpy(point_ptr + z_offset, &z, sizeof(float));
}

