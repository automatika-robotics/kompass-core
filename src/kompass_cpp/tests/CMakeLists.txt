# Set the module name
set(MODULE_NAME kompass)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Specify the location of the Python script
set(PYTHON_SCRIPT_1 ${CMAKE_CURRENT_SOURCE_DIR}/trajectory_sampler_plt.py)
set(PYTHON_SCRIPT_2 ${CMAKE_CURRENT_SOURCE_DIR}/pointcloud_scan_plt.py)
# Create a symbolic link for the Python script
execute_process(COMMAND ln -s ${PYTHON_SCRIPT_1} ${PYTHON_SCRIPT_2} ${CMAKE_CURRENT_BINARY_DIR})


# Controller Tests
add_executable(controller_test controller_test.cpp)
target_link_libraries(controller_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework nlohmann_json::nlohmann_json)
add_test(controller_tests controller_test)

# Trajectory Sampler Test (Generate and Plot Samples)
add_executable(trajectory_sampler_test trajectory_sampler_test.cpp)
target_link_libraries(trajectory_sampler_test PRIVATE ${MODULE_NAME} nlohmann_json::nlohmann_json)
add_test(trajectory_sampler_tests trajectory_sampler_test)

# Cost evaluator tests
add_executable(cost_evaluator_test cost_evaluator_test.cpp)
target_link_libraries(cost_evaluator_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
add_test(cost_evaluator_tests cost_evaluator_test)

# Mapper tests
add_executable(mapper_test mapper_test.cpp)
target_link_libraries(mapper_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
add_test(mapper_tests mapper_test)

# Pointcloud to Laserscan test
add_executable(pointcloud_test pointcloud_test.cpp)
target_link_libraries(pointcloud_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework nlohmann_json::nlohmann_json)
add_test(mapper_tests pointcloud_test)

# Collision Checker tests
add_executable(collisions_test collisions_test.cpp)
target_link_libraries(collisions_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
add_test(collisions_tests collisions_test)

# Vision DWA test
add_executable(vision_dwa_test vision_dwa_test.cpp)
target_link_libraries(vision_dwa_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework ${OpenCV_LIBS})
add_test(vision_tests vision_dwa_test)

# Vision Tracking test
add_executable(vision_tracking_test vision_tracking_test.cpp)
target_link_libraries(vision_tracking_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework nlohmann_json::nlohmann_json)
add_test(vision_tests vision_tracking_test)

# Set path to the test image
add_executable(vision_detector_test vision_detector_test.cpp)
target_link_libraries(vision_detector_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework ${OpenCV_LIBS})
add_test(vision_tests vision_detector_test)

# find sycl
find_package(AdaptiveCpp)

# add sycl based GPU tests
if(AdaptiveCpp_FOUND)
  add_executable(mapper_test_gpu mapper_test_gpu.cpp)
  target_link_libraries(mapper_test_gpu PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
  add_sycl_to_target(TARGET mapper_test_gpu SOURCES mapper_test_gpu.cpp)
  add_test(mapper_tests mapper_test_gpu)

  # Collision Checker GPU tests
  add_executable(collisions_test_gpu collisions_test_gpu.cpp)
  target_link_libraries(collisions_test_gpu PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
  add_sycl_to_target(TARGET collisions_test_gpu SOURCES collisions_test_gpu.cpp)
  add_test(collisions_tests collisions_test_gpu)

  # Add sycl to cost evaluator test if AdaptiveCpp found
  add_sycl_to_target(TARGET cost_evaluator_test SOURCES cost_evaluator_test.cpp)
endif()
