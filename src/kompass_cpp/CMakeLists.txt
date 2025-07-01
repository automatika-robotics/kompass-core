# find required packages
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(PCL COMPONENTS common io REQUIRED)
find_package(fcl REQUIRED)
find_package(ompl REQUIRED)

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
