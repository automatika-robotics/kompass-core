# find required packages
find_package(Eigen3 3.4 REQUIRED)
find_package(PCL COMPONENTS common io REQUIRED)
find_package(fcl REQUIRED)

# for bindings
set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# find sycl
find_package(AdaptiveCpp)

if(AdaptiveCpp_FOUND)
    message(STATUS "Found AdaptiveCpp. Initiating build with ACPP.")
    if(NOT ACPP_DEBUG_LEVEL)
      if(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(ACPP_DEBUG_LEVEL 3 CACHE STRING
          "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
    FORCE)
      else()
        set(ACPP_DEBUG_LEVEL 2 CACHE STRING
          "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
    FORCE)
      endif()
    endif()
    # find all sources
    file(GLOB_RECURSE SOURCES "src/*.cpp")
else()
    message(STATUS "AdaptiveCpp not found. Excluding gpu.cpp files from build.")

    # find all sources but exclude *gpu.cpp files
    file(GLOB_RECURSE ALL_SOURCES "src/*.cpp")
    set(SOURCES)
    foreach(source_file ${ALL_SOURCES})
        if(NOT "${source_file}" MATCHES "gpu\\.cpp$")
            list(APPEND SOURCES ${source_file})
        endif()
    endforeach()
endif()

# add project as library
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS} {$EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC fcl ${PCL_LIBRARIES})

# add sycl
if(AdaptiveCpp_FOUND)
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})
endif()
