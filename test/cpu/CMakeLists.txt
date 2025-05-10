set(SRC_FILES
    test_gemm.cpp
)

add_library(test_cpu_fp32 OBJECT ${SRC_FILES})
target_link_libraries(test_cpu_fp32 PUBLIC openblas gtest)
set_target_properties(test_cpu_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_library(test_cpu_fp64 OBJECT ${SRC_FILES})
# target_link_libraries(test_cpu_fp64 PUBLIC openblas gtest)
# target_compile_definitions(test_cpu_fp64 PRIVATE USE_DOUBLE)
# set_target_properties(test_cpu_fp64 PROPERTIES POSITION_INDEPENDENT_CODE ON)