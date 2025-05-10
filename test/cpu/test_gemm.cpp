#include <random>
#include <cblas.h>
#include <gtest/gtest.h>
#include "../../include/blas.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME dgemm
    #define OPENBLAS_GEMM_NAME cblas_dgemm
    #define ALLCLOSE dallclose
    #define TEST_CASE CPU_DGEMM_TEST
    #define TESTER CPU_DGEMM_TESTER
#else
    #define data_t float
    #define GEMM_NAME sgemm
    #define OPENBLAS_GEMM_NAME cblas_sgemm
    #define ALLCLOSE sallclose
    #define TEST_CASE CPU_SGEMM_TEST
    #define TESTER CPU_SGEMM_TESTER
#endif

using namespace std;


bool TESTER(int M, int N, int L) {

    std::mt19937 engine;
    std::uniform_real_distribution<data_t> dist(-1.0, 1.0);
    auto generator = [&](){return dist(engine);};

    data_t *A = (data_t*)aligned_alloc(64, N*L*sizeof(data_t));
    data_t *B = (data_t*)aligned_alloc(64, L*M*sizeof(data_t));
    data_t *C = (data_t*)aligned_alloc(64, N*M*sizeof(data_t));
    data_t *OpenBLAS_C = (data_t*)aligned_alloc(64, N*L*sizeof(data_t));

    std::generate(&A[0], &A[N*L], generator);
    std::generate(&B[0], &B[L*M], generator);

    GEMM_NAME(A, B, C, N);
    OPENBLAS_GEMM_NAME(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                        N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
    bool is_passed = ALLCLOSE(C, OpenBLAS_C, M*N);

    free(A);
    free(B);
    free(C);
    free(OpenBLAS_C);

    return is_passed;
}


TEST(TEST_CASE, same_size_matrixes) {
    int M = 1024, N = 1024, L = 1024;
    ASSERT_TRUE(TESTER(M, N, L));
}