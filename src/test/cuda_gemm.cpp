#include <iostream>
#include <gtest/gtest.h>
#include "../../include/sgemm.hpp"
#include "../../include/utils.hpp"
#include "../../include/cublas_wrapper.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME cuda_dgemm
    #define CUBLAS_GEMM_NAME cublas_dgemm
#else
    #define data_t float
    #define GEMM_NAME cuda_sgemm
    #define CUBLAS_GEMM_NAME cublas_sgemm
#endif

using namespace std;


bool _test(int M, int N, int L) {

    data_t *A = (data_t*)malloc(N*L*sizeof(data_t));
    data_t *B = (data_t*)malloc(L*M*sizeof(data_t));
    data_t *C = (data_t*)malloc(N*M*sizeof(data_t));
    data_t *cuBLAS_C = (data_t*)malloc(N*N*sizeof(data_t));

    init_vector(A, N*M);
    init_vector(B, N*M);

    GEMM_NAME(A, B, C, N);
    CUBLAS_GEMM_NAME(A, B, cuBLAS_C, N);
    bool is_passed = allclose(C, cuBLAS_C, M*N);

    free(A);
    free(B);
    free(C);
    free(cuBLAS_C);

    return is_passed;
}


TEST(CUDA_GEMM_TEST, same_size_matrixes) {
    int M = 1024, N = 1024, L = 1024;
    ASSERT_TRUE(_test(M, N, L));
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }