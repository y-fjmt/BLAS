add_subdirectory(cpu)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    add_subdirectory(cuda)
endif()

add_library(blas INTERFACE)
target_link_libraries(blas INTERFACE blas_cpu)
target_include_directories(blas INTERFACE ${CMAKE_SOURCE_DIR}/include)

if(USE_CUDA)
    target_link_libraries(blas INTERFACE blas_cuda)
endif()

