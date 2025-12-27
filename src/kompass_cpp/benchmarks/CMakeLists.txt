find_package(nlohmann_json 3.2.0 REQUIRED)

# Define the benchmark executable
add_executable(kompass_benchmark
    benchmark_runner.cpp
)

# Standard Include Directories
target_include_directories(kompass_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/kompass_cpp/include # Access to main headers
    ${PROJECT_SOURCE_DIR}/kompass_cpp/tests   # Access to test.h
)

# Link against main library
target_link_libraries(kompass_benchmark PRIVATE kompass)

# Link Third Party Deps
target_link_libraries(kompass_benchmark PRIVATE
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

# Handle AdaptiveCpp
if(AdaptiveCpp_FOUND)
    target_compile_definitions(kompass_benchmark PRIVATE GPU=1)
    add_sycl_to_target(TARGET kompass_benchmark SOURCES benchmark_runner.cpp)
endif()
