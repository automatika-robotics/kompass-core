#pragma once

#include "utils/logger.h"

#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <iostream>

// ANSI Colors for direct console output
#define BM_RESET   "\033[0m"
#define BM_CYAN    "\033[36m"
#define BM_GREEN   "\033[32m"
#define BM_YELLOW  "\033[33m"
#define BM_BOLD    "\033[1m"

using json = nlohmann::json;

namespace Kompass {
namespace Benchmarks {

struct BenchmarkResult {
    std::string test_name;
    double mean_ms;
    double std_dev_ms;
    double min_ms;
    double max_ms;
    int iterations;
};

/**
 * @brief The core benchmarking engine.
 */
template <typename Func>
BenchmarkResult measure_performance(std::string name, Func&& func, int iterations = 50, int warmup_runs = 5) {

    // --- 1. Warm-up ---
    std::stringstream ss_warm;
    ss_warm << "[Benchmark: " << name << "] Warming up (" << warmup_runs << " cycles)...";
    LOG_INFO(ss_warm.str());

    // Suppress output during warmup
    for (int i = 0; i < warmup_runs; ++i) {
        func();
    }

    // --- 2. Measurement ---
    std::vector<double> times;
    times.reserve(iterations);

    // using std::cout directly here to allow \r for in-place updates.
    std::cout << BM_CYAN << "       [Status] " << BM_RESET << "Starting " << iterations << " iterations..." << std::flush;

    // Clear the progress line with a newline
    std::cout << std::endl;

    for (int i = 0; i < iterations; ++i) {
        // Update counter every iteration
        std::cout << "\r" << BM_CYAN << "       [Status] " << BM_RESET
                  << "Iteration " << BM_BOLD << (i + 1) << "/" << iterations << BM_RESET
                  << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        func(); // Execute payload
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> ms = end - start;
        times.push_back(ms.count());
    }

    // Clear the progress line with a newline
    std::cout << std::endl;

    // --- 3. Statistics ---
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / iterations;

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double variance = (sq_sum / iterations) - (mean * mean);
    double std_dev = std::sqrt(variance > 0 ? variance : 0);

    double min_val = *std::min_element(times.begin(), times.end());
    double max_val = *std::max_element(times.begin(), times.end());

    // --- 4. Log Result ---
    std::stringstream ss_res;
    ss_res << "   -> Result: " << std::fixed << std::setprecision(3) << mean << " ms "
           << "(+/- " << std_dev << ")";
    LOG_INFO(ss_res.str());

    return {name, mean, std_dev, min_val, max_val, iterations};
}

inline void save_results_to_json(const std::string& platform_name,
                                 const std::vector<BenchmarkResult>& results,
                                 const std::string& filename) {
    json j;
    j["platform"] = platform_name;
    j["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    j["benchmarks"] = json::array();

    for (const auto& r : results) {
        j["benchmarks"].push_back({
            {"test_name", r.test_name},
            {"mean_ms", r.mean_ms},
            {"std_dev_ms", r.std_dev_ms},
            {"min_ms", r.min_ms},
            {"max_ms", r.max_ms},
            {"iterations", r.iterations}
        });
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
