cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set build type to release if not defined
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# project statement
project(
  kompass_core
  LANGUAGES C CXX)

set(PKG_DESC "Navigation Algorithms Library for Kompass")

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

# add optimization for release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# for bindings
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add kompass cpp
add_subdirectory(src/kompass_cpp/kompass_cpp)

# Add kompass cpp tests if building with cmake
if (NOT SKBUILD)
  enable_testing()
  add_subdirectory(src/kompass_cpp/tests)
endif()

# Add ompl bindings only for skbuild
if (SKBUILD)
  add_subdirectory(src/ompl)
endif()

