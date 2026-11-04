cmake_minimum_required(VERSION 3.20)
project(SegmentTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release by default for single-configuration generators (Ninja, Makefiles).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# -----------------------------------------------------------------------------
#  Sources: everything in SegmentTest/. No file list is maintained here;
#  CONFIGURE_DEPENDS makes CMake re-glob on every build, so a file added through
#  Visual Studio, Explorer or git is picked up automatically.
# -----------------------------------------------------------------------------
set(ST_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SegmentTest)

file(GLOB_RECURSE ST_SOURCES CONFIGURE_DEPENDS
    ${ST_SRC_DIR}/*.cpp
    ${ST_SRC_DIR}/*.h
    ${ST_SRC_DIR}/*.hpp
    ${ST_SRC_DIR}/*.inl)

add_executable(SegmentTest ${ST_SOURCES})
source_group(TREE ${ST_SRC_DIR} PREFIX "SegmentTest" FILES ${ST_SOURCES})

if(MSVC)
    target_compile_options(SegmentTest PRIVATE /utf-8 /W3 /permissive- /Zc:__cplusplus /MP)
    target_compile_definitions(SegmentTest PRIVATE _USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(SegmentTest PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# -----------------------------------------------------------------------------
#  CTest: run the program on data/Triangles.txt and compare with the expected
#  result for each pair ("Passed" = the pair intersects, "Failed" = it does not).
# -----------------------------------------------------------------------------
enable_testing()
add_test(NAME triangle_pairs
         COMMAND SegmentTest ${CMAKE_CURRENT_SOURCE_DIR}/data/Triangles.txt)
set_tests_properties(triangle_pairs PROPERTIES
    PASS_REGULAR_EXPRESSION
        "Test number1Passed[\r\n]+Test number3Failed[\r\n]+Test number5Failed[\r\n]+Test number7Passed[\r\n]+Test number9Failed"
    FAIL_REGULAR_EXPRESSION "Error")

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT SegmentTest)
