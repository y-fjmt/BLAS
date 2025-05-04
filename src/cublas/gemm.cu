#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../include/utils.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME cublas_dgemm
    #define CUBLAS_GEMM_NAME cublasDgemm
#else
    #define data_t float
    #define GEMM_NAME cublas_sgemm
    #define CUBLAS_GEMM_NAME cublasSgemm
#endif

void GEMM_NAME(data_t *A, data_t *B, data_t *C, size_t N, 
    float *kernel_time=nullptr) {

    data_t *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, N*N*sizeof(data_t)));
    CHECK(cudaMalloc((void**)&d_B, N*N*sizeof(data_t)));
    CHECK(cudaMalloc((void**)&d_C, N*N*sizeof(data_t)));

    CHECK(cudaMemcpy(d_A, A, N*N*sizeof(data_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, N*N*sizeof(data_t), cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));

    const data_t alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK(cudaEventRecord(start));

    CHECK_CUBLAS(
        CUBLAS_GEMM_NAME(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N)
    );

    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    cudaDeviceSynchronize();

    CHECK(cudaMemcpy(C, d_C, N*N*sizeof(data_t), cudaMemcpyDeviceToHost));

    if (kernel_time) {
        CHECK(cudaEventElapsedTime(kernel_time, start, end));
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}