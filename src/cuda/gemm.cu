#include <cuda_runtime.h>
#include "../../include/_utils.hpp"
#include "../../include/_utils_cuda.hpp"

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
#define BM 64
#define BK 8
#define TM 8
#define TN 8

namespace cuda {

    __global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

        uint thr_i = threadIdx.y;
        uint thr_j = threadIdx.x;
        uint blk_i = blockIdx.y;
        uint blk_j = blockIdx.x;

        __shared__ data_t _A[BM][BK];
        __shared__ data_t _B[BK][BN];
        
        data_t __A[TM]; 
        data_t __B[TN];
        data_t __C[TM][TN] = {0.0};

        
        for (int K = 0; K < N; K+=BK) {
            
            // load into shared memory
            uint flatten = thr_i * blockDim.x + thr_j;
            uint a_row = flatten / TM;
            uint a_col = flatten % BK;

            for (int ofs = 0; ofs < BM; ofs+=TM) {
                _A[a_row+ofs][a_col] = A[(BN*blk_i+a_row+ofs) *N+ (K+a_col)];
            }
            for (int ofs = 0; ofs < BN; ofs+=TN) {
                _B[thr_i][thr_j+ofs] = B[(K+thr_i) *N+ (blk_j*BN+thr_j+ofs)];
            }

            __syncthreads();

            for (int k = 0; k < BK; k++) {
                
                // load into register
                for (int i = 0; i < TM; i++) {
                    __A[i] = _A[thr_i*TM+i][k];
                }
                for (int j = 0; j < TN; j++) {
                    __B[j] = _B[k][thr_j*TN+j];
                }

                // compute
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        __C[i][j] += __A[i] * __B[j];
                    }
                }
                
            }
            
            __syncthreads();
        }
        
        // write back
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                C[(blk_i*BM+thr_i*TM+i) *N+ (blk_j*BN+thr_j*TN+j)] = __C[i][j];
            }
        }

    }
    

    void GEMM_NAME(data_t *A, data_t *B, data_t *C, size_t N) {
        dim3 ThreadsPerBlocks(BN/TN, BM/TM, 1);
        dim3 BlocksPerGrids(CEIL_DIV(N, BN),
                            CEIL_DIV(N, BM), 1);
        GEMM_KERNEL_NAME<<<BlocksPerGrids, ThreadsPerBlocks>>>(A, B, C, N);
    } 

}
