set(SRC_FILES
    bench_gemm.cpp
    bench_oblas_gemm.cpp
)

add_library(bench_cpu_fp32 OBJECT ${SRC_FILES})
target_link_libraries(bench_cpu_fp32 PUBLIC benchmark openblas)
set_target_properties(bench_cpu_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_library(bench_cpu_fp64 OBJECT ${SRC_FILES})
# target_link_libraries(bench_cpu_fp64 PUBLIC benchmark)
# target_compile_definitions(bench_cpu_fp64 PRIVATE USE_DOUBLE)
# set_target_properties(bench_cpu_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)