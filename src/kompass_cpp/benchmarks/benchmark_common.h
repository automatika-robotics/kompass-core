#pragma once

#include "utils/logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_POWER_MONITOR
#include <filesystem>
#include <thread>
namespace fs = std::filesystem;
#endif

// ANSI Colors for direct console output
#define BM_RESET "\033[0m"
#define BM_CYAN "\033[36m"
#define BM_GREEN "\033[32m"
#define BM_YELLOW "\033[33m"
#define BM_BOLD "\033[1m"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace Kompass {
namespace Benchmarks {

// =================================================================================
// POWER MONITOR UTILITY
// =================================================================================
#ifdef ENABLE_POWER_MONITOR
class PowerMonitor {
public:
  PowerMonitor() { detect_power_source(); }

  void start() {
    if (sensor_paths_.empty())
      return;

    running_ = true;
    readings_.clear();
    monitor_thread_ = std::thread([this]() {
      while (running_) {
        read_power();
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20Hz
      }
    });
  }

  double stop() {
    if (sensor_paths_.empty())
      return 0.0;

    running_ = false;
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }

    if (readings_.empty())
      return 0.0;

    double sum = std::accumulate(readings_.begin(), readings_.end(), 0.0);
    return sum / readings_.size(); // Average Watts
  }

  bool is_available() const { return !sensor_paths_.empty(); }

private:
  std::vector<std::string> sensor_paths_;
  std::atomic<bool> running_{false};
  std::thread monitor_thread_;
  std::vector<double> readings_;
  double scale_factor_ = 1.0;

  void detect_power_source() {
    sensor_paths_.clear();

    // ---------------------------------------------------------------------
    // STRATEGY 1: NVIDIA JETSON (INA3221)
    // ---------------------------------------------------------------------
    std::vector<std::string> jetson_candidates = {
        "/sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input", // Jetson
                                                                           // Xavier/Orin
                                                                           // NX
        "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/in1_input" // Generic
                                                                     // Orin AGX
    };

    for (const auto &path : jetson_candidates) {
      if (fs::exists(path)) {
        sensor_paths_.push_back(path);
        scale_factor_ = 1.0 / 1000.0; // mW -> W
        LOG_INFO("PowerMonitor: Found Jetson INA3221 sensor.");
        return;
      }
    }

    // ---------------------------------------------------------------------
    // STRATEGY 2: GENERIC LINUX HWMON (Prioritized for AMD/Intel)
    // ---------------------------------------------------------------------
    std::string best_candidate_path;

    if (fs::exists("/sys/class/hwmon")) {
      for (const auto &entry : fs::directory_iterator("/sys/class/hwmon")) {
        std::string hwmon_path = entry.path().string();
        std::string name_path = hwmon_path + "/name";

        // Read sensor name ("amdgpu", "zenpower", "k10temp")
        std::string sensor_name = "unknown";
        std::ifstream name_file(name_path);
        if (name_file.is_open())
          name_file >> sensor_name;

        // Look for power inputs
        for (const auto &file : fs::directory_iterator(hwmon_path)) {
          std::string filename = file.path().filename().string();

          if (filename.find("power") != std::string::npos &&
              (filename.find("_input") != std::string::npos ||
               filename.find("_average") != std::string::npos)) {

            std::ifstream p_file(file.path());
            double val;
            if (p_file >> val && val > 0) {
              // PRIORITY CHECK: AMD Strix Halo
              if (sensor_name == "amdgpu") {
                sensor_paths_.clear();
                sensor_paths_.push_back(file.path().string());
                scale_factor_ = 1.0 / 1000000.0; // microWatts
                LOG_INFO("PowerMonitor: Locked on AMD GPU/APU Sensor.");
                return;
              }

              // Keep as candidate
              best_candidate_path = file.path().string();
            }
          }
        }
      }
    }

    if (!best_candidate_path.empty()) {
      sensor_paths_.push_back(best_candidate_path);
      scale_factor_ = 1.0 / 1000000.0;
      LOG_INFO("PowerMonitor: Using Generic HWMON Sensor.");
      return;
    }

    // ---------------------------------------------------------------------
    // STRATEGY 3: POWER SUPPLY (Battery/Mains) - Rockchip
    // ---------------------------------------------------------------------
    if (fs::exists("/sys/class/power_supply")) {
      for (const auto &entry :
           fs::directory_iterator("/sys/class/power_supply")) {
        std::string p_path = entry.path().string() + "/power_now";
        if (fs::exists(p_path)) {
          std::ifstream p_file(p_path);
          double test_val;
          if (p_file >> test_val && test_val > 0) {
            sensor_paths_.push_back(p_path);
            scale_factor_ = 1.0 / 1000000.0;
            return;
          }
        }
      }
    }
  }

  void read_power() {
    double total_watts = 0.0;
    bool read_success = false;

    for (const auto &path : sensor_paths_) {
      std::ifstream file(path);
      if (file.is_open()) {
        double val;
        if (file >> val) {
          total_watts += val;
          read_success = true;
        }
      }
    }

    if (read_success) {
      readings_.push_back(total_watts * scale_factor_);
    }
  }
};

#else

// --- DUMMY IMPLEMENTATION ---
class PowerMonitor {
public:
  void start() {}               // No-op
  double stop() { return 0.0; } // Always returns 0.0
};

#endif // ENABLE_POWER_MONITOR

// =================================================================================
// BENCHMARKING ENGINE
// =================================================================================

struct BenchmarkResult {
  std::string test_name;
  double mean_ms;
  double std_dev_ms;
  double min_ms;
  double max_ms;
  int iterations;
  double avg_power_w;
};

/**
 * @brief The core benchmarking engine.
 */
template <typename Func>
BenchmarkResult measure_performance(std::string name, Func &&func,
                                    int iterations = 50, int warmup_runs = 5) {

  // Warm-up
  std::stringstream ss_warm;
  ss_warm << "[Benchmark: " << name << "] Warming up (" << warmup_runs
          << " cycles)...";
  LOG_INFO(ss_warm.str());

  for (int i = 0; i < warmup_runs; ++i) {
    func();
  }

  // Measurement
  std::vector<double> times;
  times.reserve(iterations);
  PowerMonitor power_mon;

  std::cout << BM_CYAN << "       [Status] " << BM_RESET << "Starting "
            << iterations << " iterations..." << std::flush;
  std::cout << std::endl;

  // Start Power Monitoring (different thread)
  power_mon.start();

  for (int i = 0; i < iterations; ++i) {
    std::cout << "\r" << BM_CYAN << "       [Status] " << BM_RESET
              << "Iteration " << BM_BOLD << (i + 1) << "/" << iterations
              << BM_RESET << "..." << std::flush;

    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    times.push_back(ms.count());
  }

  // Stop Power Monitoring
  double avg_watts = power_mon.stop();

  std::cout << std::endl;

  // Statistics
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / iterations;

  double sq_sum =
      std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
  double variance = (sq_sum / iterations) - (mean * mean);
  double std_dev = std::sqrt(variance > 0 ? variance : 0);

  double min_val = *std::min_element(times.begin(), times.end());
  double max_val = *std::max_element(times.begin(), times.end());

  // Log Result
  std::stringstream ss_res;
  ss_res << "   -> Result: " << std::fixed << std::setprecision(3) << mean
         << " ms "
         << "(+/- " << std_dev << ")";

  if (avg_watts > 0.0) {
    ss_res << " | Power: " << avg_watts << " W";
  }

  LOG_INFO(ss_res.str());

  return {name, mean, std_dev, min_val, max_val, iterations, avg_watts};
}

inline void save_results_to_json(const std::string &platform_name,
                                 const std::vector<BenchmarkResult> &results,
                                 const std::string &filename) {
  json j;
  j["platform"] = platform_name;
  j["timestamp"] =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  j["benchmarks"] = json::array();

  for (const auto &r : results) {
    json bench_obj = {
        {"test_name", r.test_name},   {"mean_ms", r.mean_ms},
        {"std_dev_ms", r.std_dev_ms}, {"min_ms", r.min_ms},
        {"max_ms", r.max_ms},         {"iterations", r.iterations},
    };
    if (r.avg_power_w > 0.0) {
      bench_obj["avg_power_w"] = r.avg_power_w;
    };

    j["benchmarks"].push_back(bench_obj);
  }

  std::ofstream file(filename);
  if (file.is_open()) {
    file << j.dump(4);
    file.close();
    LOG_INFO("Benchmark data saved successfully to:", filename);
  } else {
    LOG_ERROR("Unable to open JSON file for writing:", filename);
  }
}

} // namespace Benchmarks
} // namespace Kompass
