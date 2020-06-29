cmake_minimum_required(VERSION 3.4.3)

project(SyCL)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

option(SYCL_USE_OPENMP "Enable OpenMP support in samples" OFF)
option(SYCL_BUILD_TESTS "Build the tests for the header utilities in include/" OFF)

set(CMAKE_CXX_STANDARD 14)
enable_testing()

add_subdirectory(exceptions)
