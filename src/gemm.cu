#include <iostream>
#include <cstdio>
#include "../include/utils.hpp"

using namespace std;

#define data_t float

#if defined(data_t) && data_t == float
    #define GEMM_NAME cuda_sgemm
    #define GEMM_KERNEL_NAME cuda_sgemm_kernel
#elif defined(data_t) && data_t == double
    #define GEMM_NAME cuda_dgemm
    #define GEMM_KERNEL_NAME cuda_dgemm_Kernel
#else
    #error data_t is not given or invalid data type.
#endif


__global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N or j >= N) return;

    C[i*N+j] = 0.0;
    for (int k = 0; k < N; k++) {
        C[i*N+j] += A[i*N+k] * B[k*N+j];
    }

}


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

    int tpb = 16;  /* sqrt(1024) */
    dim3 ThreadsPerBlocks(tpb, tpb);
    dim3 BlocksPerGrids((N + tpb - 1) / tpb, (N + tpb - 1) / tpb);

    CHECK(cudaEventRecord(start));
    GEMM_KERNEL_NAME<<<BlocksPerGrids, ThreadsPerBlocks>>>(d_A, d_B, d_C, N);

    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    CHECK(cudaMemcpy(C, d_C, N*N*sizeof(data_t), cudaMemcpyDeviceToHost));

    if (kernel_time) {
        cudaEventElapsedTime(kernel_time, start, end);
    }
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}