# pragma once
#include <chrono>
#include "utils/logger.h"

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
