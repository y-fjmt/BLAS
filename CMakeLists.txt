cmake_minimum_required(VERSION 3.20)
project(BLAS LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

option(USE_CUDA "Enable CUDA support" OFF)

# for OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# add gtest and gbanch
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/benchmark EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/googletest EXCLUDE_FROM_ALL)

# setup OpenBLAS as external project
set(OPENBLAS_ROOT "$ENV{HOME}/local/openblas")
include_directories(${OPENBLAS_ROOT}/include)
link_directories(${OPENBLAS_ROOT}/lib)

add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(benchmark)

enable_testing()
add_test(NAME AllTests COMMAND all_tests)