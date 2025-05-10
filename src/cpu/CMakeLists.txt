set(SRC_FILES
    allclose.cpp
    gemm.cpp
)

# TODO: compatible with SSE, SSE2, and non-vector instructions
add_compile_options(-mavx512f)

add_library(blas_cpu_fp32 OBJECT ${SRC_FILES})
set_target_properties(blas_cpu_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_library(blas_cpu_fp64 OBJECT ${SRC_FILES})
# target_compile_fp64efinitions(blas_cpu_fp64 PRIVATE USE_DOUBLE)
# set_target_properties(blas_cpu_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(blas_cpu STATIC
    $<TARGET_OBJECTS:blas_cpu_fp32>
    # $<TARGET_OBJECTS:blas_cpu_fp64>
)