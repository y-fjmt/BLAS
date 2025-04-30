#include <iostream>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "../../include/utils.hpp"

using namespace std;

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME cuda_dgemm
    #define GEMM_KERNEL_NAME cuda_dgemm_Kernel
#else
    #define data_t float
    #define GEMM_NAME cuda_sgemm
    #define GEMM_KERNEL_NAME cuda_sgemm_kernel
#endif

#define BLOCK 32

__global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

    const int i = blockIdx.x * BLOCK + (threadIdx.x / BLOCK);
    const int j = blockIdx.y * BLOCK + (threadIdx.x % BLOCK);
    if (i >= N or j >= N) return;

    C[i*N+j] = 0.0;
    for (int k = 0; k < N; k++) {
        C[i*N+j] += A[i*N+k] * B[k*N+j];
    }

    // int y_global = blockIdx.y * TILE_SIZE + threadIdx.y; 
    // int x_global = blockIdx.x * TILE_SIZE + threadIdx.x; 

    // __shared__ data_t local_A[TILE_SIZE][TILE_SIZE];
    // __shared__ data_t local_B[TILE_SIZE][TILE_SIZE];

    // // load into shared memory
    // local_A[threadIdx.y][threadIdx.x] = A[y_global*N+x_global];
    // local_B[threadIdx.y][threadIdx.x] = B[y_global*N+x_global];

    // __syncthreads();

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

    dim3 ThreadsPerBlocks(BLOCK*BLOCK);
    dim3 BlocksPerGrids((N + BLOCK - 1) / BLOCK,
                        (N + BLOCK - 1) / BLOCK);

    CHECK(cudaEventRecord(start));
    GEMM_KERNEL_NAME<<<BlocksPerGrids, ThreadsPerBlocks>>>(d_A, d_B, d_C, N);

    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    CHECK(cudaMemcpy(C, d_C, N*N*sizeof(data_t), cudaMemcpyDeviceToHost));

    if (kernel_time) {
        CHECK(cudaEventElapsedTime(kernel_time, start, end));
    }
    
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
}