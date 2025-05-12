set(CMAKE_CXX_STANDARD 17)
add_subdirectory(cpu)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    add_subdirectory(cuda)
endif()

add_executable(benchmark_all
    $<TARGET_OBJECTS:bench_cpu_fp32>
    # $<TARGET_OBJECTS:bench_cpu_fp64>
    bench_main.cpp
)
target_link_libraries(benchmark_all PRIVATE 
    blas openblas benchmark
)

if(USE_CUDA)
    target_sources(benchmark_all PRIVATE
        $<TARGET_OBJECTS:bench_cuda_fp32>
        $<TARGET_OBJECTS:bench_cuda_fp64>
    )
    target_link_libraries(benchmark_all PRIVATE 
        curand cublas
    )
endif()
