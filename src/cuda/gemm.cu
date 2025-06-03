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

#define BN 64
#define BK 8
#define TM 8

namespace cuda {

    __global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

        uint thr_i = threadIdx.y;
        uint thr_j = threadIdx.x;
        uint blk_i = blockIdx.y;
        uint blk_j = blockIdx.x;

        data_t c[TM] = {0.0};
        __shared__ data_t _A[BN][BK];
        __shared__ data_t _B[BK][BN];

        for (int K = 0; K < N; K+=BK) {
            
            // load into shared memory
            _A[thr_j][thr_i] = A[(BN*blk_i+thr_j) *N+ (K+thr_i)];
            _B[thr_i][thr_j] = B[(K+thr_i) *N+ (blk_j*BN+thr_j)];

            __syncthreads();

            // compute
            for (int i = 0; i < BK; i++) {
                data_t scalar = _B[i][thr_j];
                for (int k = 0; k < TM; k++) {
                    c[k] += _A[thr_i*TM+k][i] * scalar;
                }
            }
            
            __syncthreads();
        }
        
        // write back
        for (int k = 0; k < TM; k++) {
            C[(blk_i*BN+thr_i*TM+k) *N+ (blk_j*BN+thr_j)] = c[k];
        }

    }
    

    void GEMM_NAME(data_t *A, data_t *B, data_t *C, size_t N) {
        dim3 ThreadsPerBlocks(BN, BK);
        dim3 BlocksPerGrids((N + BN - 1) / BN,
                            (N + BK - 1) / BK / TM);
        GEMM_KERNEL_NAME<<<BlocksPerGrids, ThreadsPerBlocks>>>(A, B, C, N);
    } 

}
