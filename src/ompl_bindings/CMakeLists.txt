set(MODULE_NAME ompl)

# find required packages
find_package(Eigen3 3.4 REQUIRED)
find_package(ompl 1.5 REQUIRED)

# Detect the installed nanobind package and import it into CMake
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

# add bindings as pybind11 module
nanobind_add_module(${MODULE_NAME} bindings.cpp)
target_include_directories(${MODULE_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} {$EIGEN3_INCLUDE_DIR} ${OMPL_INCLUDE_DIRS})
target_link_libraries(${MODULE_NAME} PRIVATE ${OMPL_LIBRARIES})

# install location
if (SKBUILD)
    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
endif()
