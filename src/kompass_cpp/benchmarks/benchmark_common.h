#pragma once

#include "utils/logger.h"

#include <algorithm>
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

// Only include if power monitoring is enabled
#ifdef ENABLE_POWER_MONITOR
#include <filesystem>
#include <regex>
#include <thread>
namespace fs = std::filesystem;
#endif

// ANSI Colors
#define BM_RESET "\033[0m"
#define BM_CYAN "\033[36m"
#define BM_GREEN "\033[32m"
#define BM_YELLOW "\033[33m"
#define BM_BOLD "\033[1m"

using json = nlohmann::json;

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
    if (sensors_.empty())
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
    if (sensors_.empty())
      return 0.0;
    running_ = false;
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();
    }
    if (readings_.empty())
      return 0.0;
    double sum = std::accumulate(readings_.begin(), readings_.end(), 0.0);
    return sum / readings_.size();
  }

private:
  struct PowerSensor {
    std::string path_a; // Voltage (mV) OR Direct Power
    std::string path_b; // Current (mA) OR Empty
    double scale;       // Multiplier to get Watts
  };

  std::vector<PowerSensor> sensors_;
  std::atomic<bool> running_{false};
  std::thread monitor_thread_;
  std::vector<double> readings_;

  void detect_power_source() {
    sensors_.clear();

    // -----------------------------------------------------------
    // 1. NVIDIA JETSON ORIN/AGX (INA3221 at 1-0040)
    // -----------------------------------------------------------
    // Logic: Look for 1-0040/hwmon/hwmonX.
    // Scan for in[N]_input AND curr[N]_input pairs.
    std::string jetson_base = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon";

    if (fs::exists(jetson_base)) {
      for (const auto &entry : fs::directory_iterator(jetson_base)) {
        if (entry.is_directory()) { // e.g. hwmon1
          std::string hwmon_dir = entry.path().string();
          bool found_any_rail = false;

          // Check indices 1 through 8 (INA3221 usually has 3 channels, but loop
          // safely)
          for (int i = 1; i <= 8; ++i) {
            std::string v_path =
                hwmon_dir + "/in" + std::to_string(i) + "_input";
            std::string c_path =
                hwmon_dir + "/curr" + std::to_string(i) + "_input";

            if (fs::exists(v_path) && fs::exists(c_path)) {
              // mV * mA = uW. Scale to Watts: 1e-3 * 1e-3 = 1e-6
              sensors_.push_back({v_path, c_path, 1.0e-6});
              found_any_rail = true;
            }
          }

          if (found_any_rail) {
            LOG_INFO("PowerMonitor: Found Jetson Orin Rails (V*I) at: " +
                     hwmon_dir);
            return; // Stop searching
          }
        }
      }
    }

    // -----------------------------------------------------------
    // 2. GENERIC HWMON (AMD Strix Halo / Intel - uW)
    // -----------------------------------------------------------
    std::string best_candidate_path;
    if (fs::exists("/sys/class/hwmon")) {
      for (const auto &entry : fs::directory_iterator("/sys/class/hwmon")) {
        std::string hwmon_path = entry.path().string();
        std::string name_path = hwmon_path + "/name";
        std::string sensor_name = "unknown";

        std::ifstream name_file(name_path);
        if (name_file.is_open())
          name_file >> sensor_name;

        for (const auto &file : fs::directory_iterator(hwmon_path)) {
          std::string filename = file.path().filename().string();

          if (filename.find("power") != std::string::npos &&
              (filename.find("_input") != std::string::npos ||
               filename.find("_average") != std::string::npos)) {

            std::ifstream p_file(file.path());
            double val;
            if (p_file >> val && val > 0) {
              if (sensor_name == "amdgpu") {
                sensors_.clear();
                sensors_.push_back({file.path().string(), "", 1.0 / 1000000.0});
                LOG_INFO("PowerMonitor: Locked on AMD GPU/APU Sensor (" +
                         sensor_name + ")");
                return;
              }
              best_candidate_path = file.path().string();
            }
          }
        }
      }
    }
    if (!best_candidate_path.empty()) {
      sensors_.push_back({best_candidate_path, "", 1.0 / 1000000.0});
      LOG_INFO("PowerMonitor: Using Generic HWMON Sensor");
      return;
    }

    // -----------------------------------------------------------
    // 3. POWER SUPPLY CLASS (Rockchip / RPi / Laptops)
    // -----------------------------------------------------------
    if (fs::exists("/sys/class/power_supply")) {
      for (const auto &entry :
           fs::directory_iterator("/sys/class/power_supply")) {
        std::string path_base = entry.path().string();

        // A. Direct Power (Laptops - uW)
        std::string p_path = path_base + "/power_now";
        if (fs::exists(p_path)) {
          std::ifstream pf(p_path);
          double val;
          if (pf >> val && val > 0) {
            sensors_.push_back({p_path, "", 1.0 / 1000000.0});
            LOG_INFO("PowerMonitor: Found Direct Power Supply");
            return;
          }
        }

        // B. Rockchip TCPM (Voltage * Current)
        std::string v_path = path_base + "/voltage_now";
        std::string c_path = path_base + "/current_now";

        if (fs::exists(v_path) && fs::exists(c_path)) {
          std::ifstream vf(v_path), cf(c_path);
          double v_val, c_val;
          if (vf >> v_val && cf >> c_val) {
            if (std::abs(c_val) > 0.0) {
              // uV * uA = pW. Scale to W: 1e-12
              sensors_.push_back({v_path, c_path, 1.0e-12});
              LOG_INFO("PowerMonitor: Found Rockchip/RPi Supply (V*I)");
            }
          }
        }
      }
    }
  }

  void read_power() {
    double total_watts = 0.0;
    bool success = false;

    for (const auto &sensor : sensors_) {
      std::ifstream file_a(sensor.path_a);
      double val_a = 0.0;

      if (file_a >> val_a) {
        if (sensor.path_b.empty()) {
          // Direct Power
          total_watts += val_a * sensor.scale;
          success = true;
        } else {
          // Compound (Watts = |V * I| * Scale)
          std::ifstream file_b(sensor.path_b);
          double val_b = 0.0;
          if (file_b >> val_b) {
            total_watts += std::abs(val_a * val_b) * sensor.scale;
            success = true;
          }
        }
      }
    }
    if (success)
      readings_.push_back(total_watts);
  }
};

#else
class PowerMonitor {
public:
  void start() {}
  double stop() { return 0.0; }
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

template <typename Func>
BenchmarkResult measure_performance(std::string name, Func &&func,
                                    int iterations = 50, int warmup_runs = 5) {

  std::stringstream ss_warm;
  ss_warm << "[Benchmark: " << name << "] Warming up (" << warmup_runs
          << " cycles)...";
  LOG_INFO(ss_warm.str());

  for (int i = 0; i < warmup_runs; ++i) {
    func();
  }

  std::vector<double> times;
  times.reserve(iterations);

  PowerMonitor power_mon;

  std::cout << BM_CYAN << "       [Status] " << BM_RESET << "Starting "
            << iterations << " iterations..." << std::flush;
  std::cout << std::endl;

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

  double avg_watts = power_mon.stop();

  std::cout << std::endl;

  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / iterations;

  double sq_sum =
      std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
  double variance = (sq_sum / iterations) - (mean * mean);
  double std_dev = std::sqrt(variance > 0 ? variance : 0);

  double min_val = *std::min_element(times.begin(), times.end());
  double max_val = *std::max_element(times.begin(), times.end());

  std::stringstream ss_res;
  ss_res << "   -> Result: " << std::fixed << std::setprecision(3) << mean
         << " ms " << "(+/- " << std_dev << ")";

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
        {"max_ms", r.max_ms},         {"iterations", r.iterations}};

    if (r.avg_power_w > 0.0) {
      bench_obj["avg_power_w"] = r.avg_power_w;
    }
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
