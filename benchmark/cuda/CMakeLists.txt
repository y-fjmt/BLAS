set(SRC_FILES
    bench_gemm.cu
    bench_cublas_gemm.cu
)

add_library(bench_cuda_fp32 OBJECT ${SRC_FILES})
target_link_libraries(bench_cuda_fp32 PUBLIC benchmark)
set_target_properties(bench_cuda_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(bench_cuda_fp64 OBJECT ${SRC_FILES})
target_link_libraries(bench_cuda_fp64 PUBLIC benchmark)
target_compile_definitions(bench_cuda_fp64 PRIVATE USE_DOUBLE)
set_target_properties(bench_cuda_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)