set(MODULE_NAME kompass)

# Sort sources if AdaptiveCpp found and set debug level
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
add_library(${MODULE_NAME} STATIC ${SOURCES})
target_include_directories(${MODULE_NAME} PUBLIC include)
target_include_directories(${MODULE_NAME} PUBLIC ${OMPL_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR}${FCL_INCLUDE_DIRS})
target_link_libraries(${MODULE_NAME} PUBLIC fcl ${OMPL_LIBRARIES})

# add sycl
if(AdaptiveCpp_FOUND)
    target_compile_definitions(${MODULE_NAME} PRIVATE GPU=1)
    add_sycl_to_target(TARGET ${MODULE_NAME} SOURCES ${SOURCES})
endif()

# install if built with skbuild
if (SKBUILD)
    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION .)
endif()
