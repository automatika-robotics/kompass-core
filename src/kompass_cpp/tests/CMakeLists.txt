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


# PurePursuit Controller Test
add_executable(pure_pursuit_test pure_pursuit_test.cpp)
target_link_libraries(pure_pursuit_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework nlohmann_json::nlohmann_json)
add_test(pure_pursuit_tests pure_pursuit_test)

# DWA Controller Test
add_executable(dwa_test dwa_test.cpp)
target_link_libraries(dwa_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework nlohmann_json::nlohmann_json)
add_test(dwa_tests dwa_test)

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

# Critical zone checker Checker tests
add_executable(critical_zone_test critical_zone_test.cpp)
target_link_libraries(critical_zone_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
add_test(critical_zone_tests critical_zone_test)

# Vision DWA test
add_executable(vision_dwa_test vision_dwa_test.cpp)
target_link_libraries(vision_dwa_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework ${OpenCV_LIBS})
add_test(vision_tests vision_dwa_test)

# Vision Tracking test
add_executable(vision_tracking_test vision_tracking_test.cpp)
target_link_libraries(vision_tracking_test PRIVATE ${MODULE_NAME} Boost::unit_test_framework nlohmann_json::nlohmann_json)
add_test(vision_tests vision_tracking_test)

# add sycl based GPU tests
if(AdaptiveCpp_FOUND)
  # Mapper GPU test
  add_executable(mapper_test_gpu mapper_test_gpu.cpp)
  target_link_libraries(mapper_test_gpu PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
  add_sycl_to_target(TARGET mapper_test_gpu SOURCES mapper_test_gpu.cpp)
  add_test(mapper_tests mapper_test_gpu)

  # Critical Zone Checker GPU tests
  add_executable(critical_zone_test_gpu critical_zone_test_gpu.cpp)
  target_link_libraries(critical_zone_test_gpu PRIVATE ${MODULE_NAME} Boost::unit_test_framework)
  add_sycl_to_target(TARGET critical_zone_test_gpu SOURCES critical_zone_test_gpu.cpp)
  add_test(critical_zone_tests critical_zone_test_gpu)

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
