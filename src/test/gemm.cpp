#include <iostream>
#include <cblas.h>
#include <gtest/gtest.h>
#include "../../include/gemm.hpp"
#include "../../include/utils.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME dgemm
    #define OPENBLAS_GEMM_NAME cblas_dgemm
#else
    #define data_t float
    #define GEMM_NAME sgemm
    #define OPENBLAS_GEMM_NAME cblas_sgemm
#endif

using namespace std;


bool _test(int M, int N, int L) {

    data_t *A = (data_t*)aligned_alloc(64, N*L*sizeof(data_t));
    data_t *B = (data_t*)aligned_alloc(64, L*M*sizeof(data_t));
    data_t *C = (data_t*)aligned_alloc(64, N*M*sizeof(data_t));
    data_t *OpenBLAS_C = (data_t*)aligned_alloc(64, N*L*sizeof(data_t));

    init_vector(A, N*M);
    init_vector(B, N*M);

    GEMM_NAME(A, B, C, N);
    OPENBLAS_GEMM_NAME(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                        N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
    bool is_passed = allclose(C, OpenBLAS_C, M*N);

    free(A);
    free(B);
    free(C);
    free(OpenBLAS_C);

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