# Set the module name
set(MODULE_NAME kompass)

find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)
# NOTE: The tests use the header-only Boost.Test (<boost/test/included/unit_test.hpp>), so they must NOT also link the compiled unit_test_framework library: the two
# framework copies each destroy the same singletons at exit (double free).
find_package(Boost REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Specify the location of the Python script
set(PYTHON_SCRIPT_1 ${CMAKE_CURRENT_SOURCE_DIR}/trajectory_sampler_plt.py)
set(PYTHON_SCRIPT_2 ${CMAKE_CURRENT_SOURCE_DIR}/pointcloud_scan_plt.py)
# Create a symbolic link for the Python script
execute_process(COMMAND ln -s ${PYTHON_SCRIPT_1} ${PYTHON_SCRIPT_2} ${CMAKE_CURRENT_BINARY_DIR})


# PurePursuit Controller Test
add_executable(pure_pursuit_test pure_pursuit_test.cpp)
target_link_libraries(pure_pursuit_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json)
add_test(pure_pursuit_tests pure_pursuit_test)

# DWA Controller Test
add_executable(dwa_test dwa_test.cpp)
target_link_libraries(dwa_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json)
add_test(dwa_tests dwa_test)

# Trajectory Sampler Tests (sample set checks, and plots of the samples)
add_executable(trajectory_sampler_test trajectory_sampler_test.cpp)
target_link_libraries(trajectory_sampler_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json)
add_test(trajectory_sampler_tests trajectory_sampler_test)

# Cost evaluator tests
add_executable(cost_evaluator_test cost_evaluator_test.cpp)
target_link_libraries(cost_evaluator_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json)
add_test(cost_evaluator_tests cost_evaluator_test)

# Mapper tests
add_executable(mapper_test mapper_test.cpp)
target_link_libraries(mapper_test PRIVATE ${MODULE_NAME} Boost::headers)
add_test(mapper_tests mapper_test)

# Pointcloud to Laserscan test
add_executable(pointcloud_test pointcloud_test.cpp)
target_link_libraries(pointcloud_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json)
add_test(pointcloud_tests pointcloud_test)

# Collision Checker tests
add_executable(collisions_test collisions_test.cpp)
target_link_libraries(collisions_test PRIVATE ${MODULE_NAME} Boost::headers)
add_test(collisions_tests collisions_test)

# Critical zone checker Checker tests
add_executable(critical_zone_test critical_zone_test.cpp)
target_link_libraries(critical_zone_test PRIVATE ${MODULE_NAME} Boost::headers)
add_test(critical_zone_tests critical_zone_test)

# Vision Follower test
add_executable(vision_follower_test vision_follower_test.cpp)
target_link_libraries(vision_follower_test PRIVATE ${MODULE_NAME} Boost::headers ${OpenCV_LIBS})
add_test(vision_follower_tests vision_follower_test)

# Vision Follower fixture-based parametrized test (mirrors test_vision_follower.py)
add_executable(vision_follower_fixture_test vision_follower_fixture_test.cpp)
target_link_libraries(vision_follower_fixture_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json ${OpenCV_LIBS})
add_test(vision_follower_fixture_tests vision_follower_fixture_test)

# Vision Tracking test
add_executable(vision_tracking_test vision_tracking_test.cpp)
target_link_libraries(vision_tracking_test PRIVATE ${MODULE_NAME} Boost::headers nlohmann_json::nlohmann_json)
add_test(vision_tracking_tests vision_tracking_test)

# Depth detector unit tests (point-cloud path, parity against the depth image)
add_executable(depth_detector_test depth_detector_test.cpp)
target_link_libraries(depth_detector_test PRIVATE ${MODULE_NAME} Boost::headers)
add_test(depth_detector_tests depth_detector_test)

# add sycl based GPU tests
if(AdaptiveCpp_FOUND)
  # Mapper GPU test
  add_executable(mapper_test_gpu mapper_test_gpu.cpp)
  target_link_libraries(mapper_test_gpu PRIVATE ${MODULE_NAME} Boost::headers)
  add_sycl_to_target(TARGET mapper_test_gpu SOURCES mapper_test_gpu.cpp)
  add_test(mapper_gpu_tests mapper_test_gpu)

  # Critical Zone Checker GPU tests
  add_executable(critical_zone_test_gpu critical_zone_test_gpu.cpp)
  target_link_libraries(critical_zone_test_gpu PRIVATE ${MODULE_NAME} Boost::headers)
  add_sycl_to_target(TARGET critical_zone_test_gpu SOURCES critical_zone_test_gpu.cpp)
  add_test(critical_zone_gpu_tests critical_zone_test_gpu)

  # NOTE: Add private GPU flag since these targets interacts with parts of
  # the library where code is mixed and this flag is used. We don't make the
  # flag public in the library since we don't want to force all downstream targets
  # to carry it, but we do need it for this test.
  # Cost evaluator test
  target_compile_definitions(cost_evaluator_test PRIVATE GPU=1)
  add_sycl_to_target(TARGET cost_evaluator_test SOURCES cost_evaluator_test.cpp)

  # Controller Tests
  target_compile_definitions(pure_pursuit_test PRIVATE GPU=1)
  add_sycl_to_target(TARGET pure_pursuit_test SOURCES pure_pursuit_test.cpp)

  target_compile_definitions(dwa_test PRIVATE GPU=1)
  add_sycl_to_target(TARGET dwa_test SOURCES dwa_test.cpp)
endif()
