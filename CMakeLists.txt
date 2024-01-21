cmake_minimum_required(VERSION 3.26)

project(nykdtb
    VERSION 0.1
    DESCRIPTION "Base library for hobby development"
    LANGUAGES CXX
)

include(cmake/defines.cmake)
include(cmake/configure-compiler-warnings.cmake)
include(cmake/configure-instrumentations.cmake)

if((PROJECT_IS_TOP_LEVEL OR NYKDTB_BUILD_TESTING))
    include(CTest)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(targets)

if((PROJECT_IS_TOP_LEVEL OR NYKDTB_BUILD_TESTING) AND BUILD_TESTING)
    add_subdirectory(tests)
endif()