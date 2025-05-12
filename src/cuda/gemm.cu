#include <cuda_runtime.h>
#include "../../include/_utils.hpp"

using namespace std;

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME dgemm
    #define GEMM_KERNEL_NAME dgemm_kernel
#else
    #define data_t float
    #define GEMM_NAME sgemm
    #define GEMM_KERNEL_NAME sgemm_kernel
#endif

#define TS_M 32
#define TS_N 32
#define TS_L 32

namespace cuda {

    __global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

        int M = N;
        // int L = N;

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
            _A[i][j] = A[(I+i)*N + (K+j)];
            _B[i][j] = B[(K+i)*N + (J+j)];

            __syncthreads();

            // compute sub matrix
            for (int k = 0; k < TS_L; k++) {
                c += _A[i][k] * _B[k][j];
            }

            __syncthreads();
        }

        C[(I+i) * N + (J+j)] = c;

    }
    

    void GEMM_NAME(data_t *A, data_t *B, data_t *C, size_t N) {
        dim3 ThreadsPerBlocks(TS_M, TS_N);
        dim3 BlocksPerGrids((N + TS_M - 1) / TS_M,
                            (N + TS_N - 1) / TS_N);
        GEMM_KERNEL_NAME<<<BlocksPerGrids, ThreadsPerBlocks>>>(A, B, C, N);
    } 

}
