# Set the module name
set(MODULE_NAME kompass_cpp)

# for bindings
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Take out GPU bindings if adaptive cpp not found
if(NOT AdaptiveCpp_FOUND)
    file(GLOB_RECURSE ALL_SOURCES "*.cpp")
    set(SOURCES)
    foreach(source_file ${ALL_SOURCES})
        if(NOT "${source_file}" MATCHES "gpu\\.cpp$")
            list(APPEND SOURCES ${source_file})
        endif()
    endforeach()
else()
    file(GLOB_RECURSE SOURCES "*.cpp")
endif()

# add project as pybind module
pybind11_add_module(${MODULE_NAME} MODULE ${SOURCES})
target_link_libraries(${MODULE_NAME} PRIVATE kompass)

if (AdaptiveCpp_FOUND)
    target_compile_definitions(${MODULE_NAME} PRIVATE GPU=1)
    add_sycl_to_target(TARGET ${MODULE_NAME} SOURCES ${SOURCES})
endif()

# install location
install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
