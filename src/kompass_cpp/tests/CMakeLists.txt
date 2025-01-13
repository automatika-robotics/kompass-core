find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

# Controller Tests
add_executable(controller_test controller_test.cpp)
target_link_libraries(controller_test PRIVATE ${PROJECT_NAME} Boost::unit_test_framework)
add_test(controller_tests controller_test)

# Find Python interpreter
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Specify the location of the Python script
set(PYTHON_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/trajectory_sampler_plt.py)

# Create a symbolic link or copy to create a "binary" for the Python script
add_custom_target(trajectory_sampler_plt ALL
    DEPENDS ${PYTHON_SCRIPT}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${PYTHON_SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/trajectory_sampler_plt
)


# Trajectory Sampler Test (Generate and Plot Samples)
add_executable(trajectory_sampler_test trajectory_sampler_test.cpp)
target_link_libraries(trajectory_sampler_test PRIVATE ${PROJECT_NAME} nlohmann_json::nlohmann_json)
add_test(trajectory_sampler_tests trajectory_sampler_test)

# Mapper tests
add_executable(mapper_test mapper_test.cpp)
target_link_libraries(mapper_test PRIVATE ${PROJECT_NAME} Boost::unit_test_framework)
add_test(mapper_tests mapper_test)

# find sycl
find_package(AdaptiveCpp)

# add sycl based GPU tests
if(AdaptiveCpp_FOUND)
  add_executable(mapper_test_gpu mapper_test_gpu.cpp)
  target_link_libraries(mapper_test_gpu PRIVATE ${PROJECT_NAME} Boost::unit_test_framework)
  add_sycl_to_target(TARGET mapper_test_gpu SOURCES mapper_test_gpu.cpp)
  add_test(mapper_tests mapper_test_gpu)
endif()

