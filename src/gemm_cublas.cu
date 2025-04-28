#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(data_t) && data_t == float
    #define CUBLAS_GEMM_FN cublasSgemm
    #define CUBLAS_GEMM cuda_blas_sgemm
#elif defined(data_t) && data_t == double
    #define CUBLAS_GEMM_FN cublasDgemm
    #define CUBLAS_GEMM cuda_blas_dgemm
#else
    #error data_t is not given or invalid data type.
#endif


void CUBLAS_GEMM(data_t* A, data_t* B, data_t* C, size_t N,
            float *kernel_time=nullptr) {

    data_t *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N*sizeof(data_t));
    cudaMalloc((void**)&d_B, N*N*sizeof(data_t));
    cudaMalloc((void**)&d_C, N*N*sizeof(data_t));
    cudaMemcpy(d_A, A, N*N*sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(data_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cublasHandle_t handle;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cublasCreate(&handle);

    const data_t alpha = 1.0;
    const data_t beta = 0.0;

    cudaEventRecord(start);

    CUBLAS_GEMM_FN(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        A, N,
        B, N,
        &beta,
        C, N
    );

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaMemcpy(d_C, C, N*N*sizeof(data_t), cudaMemcpyDeviceToHost);

    if (kernel_time) {
        cudaEventElapsedTime(kernel_time, start, end);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}