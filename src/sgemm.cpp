#include <omp.h>
#include <immintrin.h>

#define B_L1 64
#define B_L2 256
#define B_L3 512


void _my_sgemm_comp(float* A, float* B, float* C, int N, int I, int J, int K) {
    __m512 zmm1, zmm2, zmm_accum;
    for (int i = I; i < I+B_L1; i++) {
        for (int j = J; j < J+B_L1; j+=16) {
            zmm_accum = _mm512_load_ps(&C[i*N+j]);
            // for (int k = K; k < K+B_L1; k++) {
            //     zmm1 = _mm512_set1_ps(A[i*N+k]);
            //     zmm2 = _mm512_load_ps(&B[k*N+j]);
            //     zmm_accum = _mm512_fmadd_ps(zmm1, zmm2, zmm_accum);
            // }

            for (int k = K; k < K+B_L1; k+=16) {

            #define CALC(X) \
                zmm1 = _mm512_set1_ps(A[i*N+(k+(X))]); \
                zmm2 = _mm512_load_ps(&B[(k+(X))*N+j]); \
                zmm_accum = _mm512_fmadd_ps(zmm1, zmm2, zmm_accum);

                CALC(0)
                CALC(1)
                CALC(2)
                CALC(3)
                CALC(4)
                CALC(5)
                CALC(6)
                CALC(7)
                CALC(8)
                CALC(9)
                CALC(10)
                CALC(11)
                CALC(12)
                CALC(13)
                CALC(14)
                CALC(15)
            }

            _mm512_store_ps(&C[i*N+j], zmm_accum);
        }
    }
}


void _my_sgemm_l1(float* A, float* B, float* C, int N, int I, int J, int K) {
    for (int i = I; i < I+B_L2; i+=B_L1) {
        for (int j = J; j < J+B_L2; j+=B_L1) {
            for (int k = K; k < K+B_L2; k+=B_L1) {
                _my_sgemm_comp(A, B, C, N, i, j, k);
            }
        }
    }
}


void _my_sgemm_l2(float* A, float* B, float* C, int N, int I, int J, int K) {
    for (int i = I; i < I+B_L3; i+=B_L2) {
        for (int j = J; j < J+B_L3; j+=B_L2) {
            for (int k = K; k < K+B_L3; k+=B_L2) {
                _my_sgemm_l1(A, B, C, N, i, j, k);
            }
        }
    }
}


void _my_sgemm_l3(float* A, float* B, float* C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i+=B_L3) {
        for (int j = 0; j < N; j+=B_L3) {
            for (int k = 0; k < N; k+=B_L3) {
                _my_sgemm_l2(A, B, C, N, i, j, k);
            }
        }
    }
}


void my_sgemm(float* A, float* B, float* C, int N) {
    _my_sgemm_l3(A, B, C, N);
}