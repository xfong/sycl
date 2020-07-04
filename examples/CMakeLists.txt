cmake_minimum_required(VERSION 3.4.3)

project(SyCL)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

option(SYCL_USE_OPENMP "Enable OpenMP support in samples" OFF)
option(SYCL_BUILD_TESTS "Build the tests for the header utilities in include/" OFF)

set(CMAKE_CXX_STANDARD 14)
enable_testing()

add_subdirectory(custom_device_selector)
add_subdirectory(dotproduct)
add_subdirectory(exceptions)
add_subdirectory(get_device_queue)
add_subdirectory(madd2)
add_subdirectory(madd3)
add_subdirectory(parfor_encrypt)
add_subdirectory(reduce_sum)
add_subdirectory(reduce_sample)
add_subdirectory(simple_vec_add)
add_subdirectory(template_func)
