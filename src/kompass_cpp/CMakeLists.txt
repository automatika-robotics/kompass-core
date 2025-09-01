# find required packages
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(fcl 0.7 REQUIRED)
find_package(ompl 1.5 REQUIRED)

# find sycl
find_package(AdaptiveCpp)

# Build static lib for kompass cpp
add_subdirectory(kompass_cpp)

# Add kompass cpp tests if building with cmake
if (NOT SKBUILD)
  enable_testing()
  add_subdirectory(tests)
endif()

# Add bindings if skbuild
if (SKBUILD OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_subdirectory(bindings)
endif()
