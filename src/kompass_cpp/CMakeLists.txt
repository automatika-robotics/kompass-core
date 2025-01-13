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
  kompass_cpp
  LANGUAGES C CXX)

set(PKG_DESC "KOMPASS CPP Robot Motion Control and Planning Library")

enable_testing()

add_subdirectory(${PROJECT_NAME})
add_subdirectory(tests)
