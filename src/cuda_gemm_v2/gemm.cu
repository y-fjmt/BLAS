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

#define TS_M 32
#define TS_N 32
#define TS_L 32

__global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

    int M = N;
    int L = N;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= N) return;

    __shared__ data_t _A[TS_M][TS_L];
    __shared__ data_t _B[TS_L][TS_N];
    
    data_t c = 0.0;

    i = threadIdx.y;
    j = threadIdx.x;

    int I = blockIdx.y * TS_M;
    int J = blockIdx.x * TS_N;

    for (int K = 0; K < N; K+=TS_L) {
        
        // load into shared memory        
        // FIXME: (K+j),(K+i)は正方行列限定
        _A[i][j] = A[(I+i)*N + (K+j)];
        _B[i][j] = B[(K+i)*N + (J+j)];

        // Coalescing
        // _B[i][j] = B[(J+j)*N]+(K+i);

        __syncthreads();

        // compute sub matrix
        for (int k = 0; k < TS_L; k++) {
            c += _A[i][k] * _B[k][j];
        }

        __syncthreads();
    }

    C[(I+i) * N + (J+j)] = c;

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

    dim3 ThreadsPerBlocks(TS_M, TS_N);
    dim3 BlocksPerGrids((N + TS_M - 1) / TS_M,
                        (N + TS_N - 1) / TS_N);

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