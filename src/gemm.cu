#include <iostream>
#include <cstdio>

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

    if (i == 0 and j == 0) {
        printf("%f\n", C[i*N+j]);
    }

}


void GEMM_NAME(data_t *A, data_t *B, data_t *C, size_t N, 
                float *kernel_time=nullptr) {

    data_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N*sizeof(data_t));
    cudaMalloc((void**)&d_B, N*N*sizeof(data_t));
    cudaMalloc((void**)&d_C, N*N*sizeof(data_t));

    cudaMemcpy(d_A, A, N*N*sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(data_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int threads_1d = 1024;
    int blocks_1d = (N + threads_1d - 1) / threads_1d;
    dim3 threads(threads_1d, threads_1d);
    dim3 blocks(blocks_1d, blocks_1d);

    cudaEventRecord(start);
    GEMM_KERNEL_NAME<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();

    cudaMemcpy(d_C, C, N*N*sizeof(data_t), cudaMemcpyDeviceToHost);

    if (kernel_time) {
        cudaEventElapsedTime(kernel_time, start, end);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}