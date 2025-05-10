set(CMAKE_CXX_STANDARD 17)
add_subdirectory(cpu)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_subdirectory(cuda)
endif()

add_executable(all_tests
    $<TARGET_OBJECTS:test_cpu_fp32>
    # $<TARGET_OBJECTS:test_cpu_fp64>
    test_main.cpp
)
target_link_libraries(all_tests PRIVATE 
    blas openblas
    gtest gtest_main pthread
)

if(USE_CUDA)
    target_sources(all_tests PRIVATE
        $<TARGET_OBJECTS:test_cuda_fp32>
        $<TARGET_OBJECTS:test_cuda_fp64>
    )
    target_link_libraries(all_tests PRIVATE 
    curand cublas
)
endif()
