#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "../../include/blas_cuda.hpp"
#include "../../include/_utils_cuda.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM cuda::dgemm
    #define GEMM_CUBLAS cublasDgemm
    #define RANDOM curandGenerateUniformDouble
    #define ALLCLOSE cuda::dallclose
    #define TEST_CASE CUDA_DGEMM_TEST
    #define TESTER CUDA_DGEMM_TESTER
#else
    #define data_t float
    #define GEMM cuda::sgemm
    #define GEMM_CUBLAS cublasSgemm
    #define RANDOM curandGenerateUniform
    #define ALLCLOSE cuda::sallclose
    #define TEST_CASE CUDA_SGEMM_TEST
    #define TESTER CUDA_SGEMM_TESTER
#endif


bool TESTER(int M, int N, int K) {

    curandGenerator_t gen;

    data_t *A, *B, *C, *cublas_C;
    CUDA_CALL(cudaMalloc((void**)&A, N*K*sizeof(data_t)));
    CUDA_CALL(cudaMalloc((void**)&B, K*M*sizeof(data_t)));
    CUDA_CALL(cudaMalloc((void**)&C, N*M*sizeof(data_t)));
    CUDA_CALL(cudaMalloc((void**)&cublas_C, N*M*sizeof(data_t)));

    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CALL(RANDOM(gen, A, N*K));
    CURAND_CALL(RANDOM(gen, B, K*M));
    CURAND_CALL(RANDOM(gen, C, N*M));
    CURAND_CALL(RANDOM(gen, cublas_C, N*M));
    
    GEMM(A, B, C, N);

    data_t alpha = 1.0f;
    data_t beta = 0.0f;
    cublasHandle_t handle; 
    cublasCreate(&handle);

    GEMM_CUBLAS(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            A, N,
            B, N,
            &beta,
            cublas_C, N);

    cudaDeviceSynchronize();
    bool is_passed = ALLCLOSE(C, cublas_C, N*M);


    cudaDeviceSynchronize();

    CURAND_CALL(curandDestroyGenerator(gen));
    cublasDestroy(handle);
    CUDA_CALL(cudaFree(A));
    CUDA_CALL(cudaFree(B));
    CUDA_CALL(cudaFree(C));
    CUDA_CALL(cudaFree(cublas_C));

    return is_passed;
}


TEST(TEST_CASE, same_size_matrixes) {
    int M = 1024, N = 1024, L = 1024;
    ASSERT_TRUE(TESTER(M, N, L));
}