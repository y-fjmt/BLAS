set(SRC_FILES
    allclose.cu
    gemm.cu
)

add_library(blas_cuda_fp32 OBJECT ${SRC_FILES})
set_target_properties(blas_cuda_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(blas_cuda_fp64 OBJECT ${SRC_FILES})
target_compile_definitions(blas_cuda_fp64 PRIVATE USE_DOUBLE)
set_target_properties(blas_cuda_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(blas_cuda STATIC
    $<TARGET_OBJECTS:blas_cuda_fp32>
    $<TARGET_OBJECTS:blas_cuda_fp64>
)