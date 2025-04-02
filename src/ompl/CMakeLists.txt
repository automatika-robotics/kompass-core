set(MODULE_NAME ompl)

# find required packages
find_package(Eigen3 3.4 REQUIRED)
find_package(ompl 1.5 REQUIRED)

# add project as library
pybind11_add_module(${MODULE_NAME} MODULE src/bindings.cpp)
target_include_directories(${MODULE_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} {$EIGEN3_INCLUDE_DIR} ${OMPL_INCLUDE_DIRS})
target_link_libraries(${MODULE_NAME} PRIVATE ${OMPL_LIBRARIES})

# install location
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
