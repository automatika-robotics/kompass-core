# find required packages
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(fcl 0.7 REQUIRED)
find_package(ompl 1.5 REQUIRED)

# An option to force CPU build, default is OFF (allow GPU if found)
option(FORCE_CPU_BUILD "Force CPU compilation even if AdaptiveCpp is available" OFF)

# find sycl ONLY if the flag is not set
if(NOT FORCE_CPU_BUILD)
    find_package(AdaptiveCpp)
endif()

# Build static lib for kompass cpp
add_subdirectory(kompass_cpp)

# Add kompass cpp tests if building with cmake
if (NOT SKBUILD)
  enable_testing()
  add_subdirectory(tests)
endif()

# Add bindings if skbuild or debugging
if (SKBUILD OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_subdirectory(bindings)
endif()

# Build benchmarking if building with cmake and not debugging
if (NOT SKBUILD AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_subdirectory(benchmarks)
endif()

