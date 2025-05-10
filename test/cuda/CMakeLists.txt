set(SRC_FILES
    test_gemm.cu
)

add_library(test_cuda_fp32 OBJECT ${SRC_FILES})
target_link_libraries(test_cuda_fp32 PUBLIC gtest curand cublas)
set_target_properties(test_cuda_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(test_cuda_fp64 OBJECT ${SRC_FILES})
target_link_libraries(test_cuda_fp64 PUBLIC gtest curand cublas)
target_compile_definitions(test_cuda_fp64 PRIVATE USE_DOUBLE)
set_target_properties(test_cuda_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)