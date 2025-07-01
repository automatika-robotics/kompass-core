#pragma once

#include <string>

/**
 * @brief Retrieves the names of SYCL-compatible accelerator devices available
 * on the system.
 *
 * This function queries the system using SYCL's runtime API to list the names
 * of all devices across all platforms (e.g., GPUs, CPUs, and accelerators). The
 * function only executes if the GPU macro is defined at compile time (e.g., via
 * -DGPU=1 or target_compile_definitions in CMake). If the GPU macro is not
 * defined, or no devices are available, an empty string is returned.
 *
 * @return A newline-separated string of device names, or an empty string if GPU
 * support is not enabled or no devices are found.
 */
std::string getAvailableAccelerators();
