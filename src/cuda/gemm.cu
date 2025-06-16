#include <cuda_runtime.h>
#include "../../include/_utils.hpp"
#include "../../include/_utils_cuda.hpp"

using namespace std;

#if defined(USE_DOUBLE)
    #define data_t double
    #define vec_data_t double2
    #define GEMM_NAME dgemm
    #define GEMM_KERNEL_NAME dgemm_kernel
#else
    #define data_t float
    #define vec_data_t float4
    #define GEMM_NAME sgemm
    #define GEMM_KERNEL_NAME sgemm_kernel
#endif

#define BN 64
#define BM 64
#define BK 8
#define TM 8
#define TN 8

constexpr uint n_thrs = (BN / TN * BM / TM);
constexpr uint vec_len = sizeof(vec_data_t) / sizeof(data_t);

namespace cuda {

    __global__ void GEMM_KERNEL_NAME(data_t *A, data_t *B, data_t *C, size_t N) {

        const uint a_row = threadIdx.x / (BK / vec_len);
        const uint a_col = threadIdx.x % (BK / vec_len);
        const uint b_row = threadIdx.x / (BN / TN);
        const uint b_col = threadIdx.x % (BN / TN);
        
        __shared__ data_t _A[BK][BN+1];
        __shared__ data_t _B[BK][BN];
        
        data_t __A[TM]; 
        data_t __B[TN];
        data_t __C[TM*TN] = {0.0};

        A += (BN * blockIdx.y) * N;
        B += blockIdx.x * BN;

        for (int K = 0; K < N; K+=BK) {
            
            // load partial A into shared memory
            for (int ofs = 0; ofs < BM; ofs+=(n_thrs / (BK / vec_len))) {
                vec_data_t tmp = *(reinterpret_cast<vec_data_t*>(&A[(ofs + a_row) *N+ (a_col * vec_len)]));

                #if defined(USE_DOUBLE)
                    _A[a_col * vec_len + 0][ofs + a_row] = tmp.x;
                    _A[a_col * vec_len + 1][ofs + a_row] = tmp.y;
                #else
                    _A[a_col * vec_len + 0][ofs + a_row] = tmp.x;
                    _A[a_col * vec_len + 1][ofs + a_row] = tmp.y;
                    _A[a_col * vec_len + 2][ofs + a_row] = tmp.z;
                    _A[a_col * vec_len + 3][ofs + a_row] = tmp.w;
                #endif
            }

            // load partial B into shared memory
            for (int ofs = 0; ofs < BN; ofs+=(TN*vec_len)) {
                *(reinterpret_cast<vec_data_t*>(&_B[b_row][ofs + b_col * vec_len])) = \
                    *(reinterpret_cast<vec_data_t*>(&B[(b_row) *N+ (ofs + b_col * vec_len)]));
            }
        
            A += BK;
            B += BK * N;
            __syncthreads();

            for (int k = 0; k < BK; k++) {
                
                // load into register
                for (int i = 0; i < TM; i++) {
                    __A[i] = _A[k][b_row*TM+i];
                }
                for (int j = 0; j < TN; j++) {
                    __B[j] = _B[k][b_col*TN+j];
                }

                // compute
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        __C[i*TM+j] += __A[i] * __B[j];
                    }
                }
                
            }
            
            __syncthreads();
        }

        C += (blockIdx.y * BM + b_row * TM) * N + (blockIdx.x * BN + b_col * TN);
        
        // write back
        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
                C[i * N+ j] = __C[i * TM + j];
            }
        }

    }
    

    void GEMM_NAME(data_t *A, data_t *B, data_t *C, size_t N) {
        dim3 ThreadsPerBlocks((BN / TN) * BM / TM, 1);
        dim3 BlocksPerGrids(CEIL_DIV(N, BN),
                            CEIL_DIV(N, BM), 1);
        GEMM_KERNEL_NAME<<<BlocksPerGrids, ThreadsPerBlocks>>>(A, B, C, N);
    } 

}
