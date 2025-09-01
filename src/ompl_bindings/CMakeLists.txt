set(MODULE_NAME omplpy)

# find required packages
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(ompl 1.5 REQUIRED)

# Detect nanobind (injected by scikit build)
find_package(nanobind CONFIG)

# Fallback: query Python for nanobind’s cmake_dir if not found
if(NOT nanobind_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_DIR
        RESULT_VARIABLE nanobind_result
    )
    if(nanobind_result EQUAL 0 AND EXISTS "${nanobind_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
        find_package(nanobind CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "nanobind not found. Please pip install nanobind.")
    endif()
endif()

# add bindings as nanobind module
nanobind_add_module(${MODULE_NAME} STABLE_ABI bindings.cpp)
target_include_directories(${MODULE_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR} ${OMPL_INCLUDE_DIRS})
target_link_libraries(${MODULE_NAME} PRIVATE ${OMPL_LIBRARIES})

# install location
if (SKBUILD)
    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
endif()
