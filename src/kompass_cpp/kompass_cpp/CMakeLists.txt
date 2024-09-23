# find required packages
find_package(Eigen3 3.4 REQUIRED)
find_package(PCL REQUIRED)
find_package(fcl REQUIRED)

# for bindings
set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# find all sources
file(GLOB_RECURSE SOURCES "src/*.cpp")

# add project as library
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS} {$EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC fcl ${PCL_LIBRARIES})
