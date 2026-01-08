find_package(nlohmann_json 3.2.0 REQUIRED)

# Toggles power monitoring
option(ENABLE_POWER_MONITOR "Enable power monitoring (may affect benchmark accuracy)" OFF)

# Define the benchmark executable
add_executable(kompass_benchmark
    benchmark_runner.cpp
)

# Apply Feature Flags
if(ENABLE_POWER_MONITOR)
    message(STATUS "Benchmark Power Monitoring: ENABLED (Threaded polling active)")
    target_compile_definitions(kompass_benchmark PRIVATE ENABLE_POWER_MONITOR)
else()
    message(STATUS "Benchmark Power Monitoring: DISABLED (Pure performance mode)")
endif()

# Standard Include Directories
target_include_directories(kompass_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/kompass_cpp/include
    ${PROJECT_SOURCE_DIR}/kompass_cpp/tests
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
