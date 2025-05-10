#include <omp.h>
#include <immintrin.h>

// Optimized for 2 x Intel(R) Xeon(R) Platinum 8468
#define B_L1 32
#define B_L2 128
#define B_L3 256
#define SIMD_LEN 16

inline void _order(float* A, float* B, float* C, float* _A, float* _B, float* _C, int N);
inline void _compute(float* _A, float* _B, float* _C, int N);
inline void _reorder(float* C, float *_C, int N);


void sgemm(float* A, float* B, float* C, int N) {

    float *_A = (float*)aligned_alloc(64, N*N*sizeof(float));
    float *_B = (float*)aligned_alloc(64, N*N*sizeof(float));
    float *_C = (float*)aligned_alloc(64, N*N*sizeof(float));

    #pragma omp parallel 
    {
        _order(A, B, C, _A, _B, _C, N);
        #pragma omp barrier
        _compute(_A, _B, _C, N);
        #pragma omp barrier
        _reorder(C, _C, N);
    }

    free(_A);
    free(_B);
    free(_C);
}


inline void _order(float* A, float* B, float* C, float* _A, float* _B, float* _C, int N) {

    __m512 zmm_cpy, zmm_zeros;
    zmm_zeros = _mm512_setzero_ps();

    #pragma omp for private(zmm_cpy), collapse(2)
    for (int i_l3 = 0; i_l3 < N; i_l3+=B_L3) {
        for (int j_l3 = 0; j_l3 < N; j_l3+=B_L3) {

            int idx = (i_l3 * (N/B_L3) + j_l3) * B_L3;
            
            for (int i_l2 = i_l3; i_l2 < i_l3+B_L3; i_l2+=B_L2) {
                for (int j_l2 = j_l3; j_l2 < j_l3+B_L3; j_l2+=B_L2) {

                    for (int i_l1 = i_l2; i_l1 < i_l2+B_L2; i_l1+=B_L1) {
                        for (int j_l1 = j_l2; j_l1 < j_l2+B_L2; j_l1+=B_L1) {

                            int _idx = idx;
                            for (int i = i_l1; i < i_l1+B_L1; i++) {
                                for (int j = j_l1; j < j_l1+B_L1; j+=SIMD_LEN) {
                                    zmm_cpy = _mm512_load_ps(&A[i*N+j]);
                                    _mm512_store_ps(&_A[_idx], zmm_cpy);
                                    _mm512_store_ps(&_C[_idx], zmm_zeros);
                                    _idx+=SIMD_LEN;
                                }
                            }

                            for (int i = j_l1; i < j_l1+B_L1; i++) {
                                for (int j = i_l1; j < i_l1+B_L1; j+=SIMD_LEN) {
                                    zmm_cpy = _mm512_load_ps(&B[i*N+j]);
                                    _mm512_store_ps(&_B[idx], zmm_cpy);
                                    idx+=16;
                                }
                            }

                        }
                    }

                }
            }
            
        }
    }
}


inline void _compute(float* _A, float* _B, float* _C, int N) {

    int N_L3 =    N / B_L3, B_L3_2 = B_L3 * B_L3;
    int N_L2 = B_L3 / B_L2, B_L2_2 = B_L2 * B_L2;
    int N_L1 = B_L2 / B_L1, B_L1_2 = B_L1 * B_L1;

    // L3 Blocks
    #pragma omp for collapse(2)
    for (int i_l3 = 0; i_l3 < N_L3; i_l3++) {
        for (int j_l3 = 0; j_l3 < N_L3; j_l3++) {
            for (int k_l3 = 0; k_l3 < N_L3; k_l3++) {

                float *_A_L3 = _A + ((i_l3 * N_L3) + k_l3) * B_L3_2;
                float *_B_L3 = _B + ((j_l3 * N_L3) + k_l3) * B_L3_2;
                float *_C_L3 = _C + ((i_l3 * N_L3) + j_l3) * B_L3_2;

                // L2 Blocks
                for (int i_l2 = 0; i_l2 < N_L2; i_l2++) {
                    for (int j_l2 = 0; j_l2 < N_L2; j_l2++) {
                        for (int k_l2 = 0; k_l2 < N_L2; k_l2++) {
                            
                            float *_A_L2 = _A_L3 + ((i_l2 * N_L2) + k_l2) * B_L2_2; 
                            float *_B_L2 = _B_L3 + ((j_l2 * N_L2) + k_l2) * B_L2_2; 
                            float *_C_L2 = _C_L3 + ((i_l2 * N_L2) + j_l2) * B_L2_2; 

                            // L1 Blocks
                            for (int i_l1 = 0; i_l1 < N_L1; i_l1++) {
                                for (int j_l1 = 0; j_l1 < N_L1; j_l1++) {
                                    for (int k_l1 = 0; k_l1 < N_L1; k_l1++) {

                                        float *_A_L1 = _A_L2 + ((i_l1 * N_L1) + k_l1) * B_L1_2;
                                        float *_B_L1 = _B_L2 + ((j_l1 * N_L1) + k_l1) * B_L1_2;
                                        float *_C_L1 = _C_L2 + ((i_l1 * N_L1) + j_l1) * B_L1_2;

                                        __m512 zmm1, zmm2, zmm_accum;
                                        
                                        // computation
                                        for (int i = 0; i < B_L1; i++) {
                                            for (int j = 0; j < B_L1; j+=SIMD_LEN) {

                                                zmm_accum = _mm512_load_ps(&_C_L1[i*B_L1+j]);

                                                for (int k = 0; k < B_L1; k+=16) {
                                                #define CALC(X) \
                                                    zmm1 = _mm512_set1_ps(_A_L1[i*B_L1+k+(X)]); \
                                                    zmm2 = _mm512_load_ps(&_B_L1[(k+(X))*B_L1+j]); \
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

                                                _mm512_store_ps(&_C_L1[i*B_L1+j], zmm_accum);
                                            }
                                        }

                                    }
                                }
                            }

                        }
                    }
                }

            }
        }
    }

}


inline void _reorder(float* C, float *_C, int N) {

    __m512 zmm_cpy;

    #pragma omp for private(zmm_cpy), collapse(2)
    for (int i_l3 = 0; i_l3 < N; i_l3+=B_L3) {
        for (int j_l3 = 0; j_l3 < N; j_l3+=B_L3) {

            int idx = (i_l3 * (N/B_L3) + j_l3) * B_L3;
            
            for (int i_l2 = i_l3; i_l2 < i_l3+B_L3; i_l2+=B_L2) {
                for (int j_l2 = j_l3; j_l2 < j_l3+B_L3; j_l2+=B_L2) {

                    for (int i_l1 = i_l2; i_l1 < i_l2+B_L2; i_l1+=B_L1) {
                        for (int j_l1 = j_l2; j_l1 < j_l2+B_L2; j_l1+=B_L1) {

                            for (int i = i_l1; i < i_l1+B_L1; i++) {
                                for (int j = j_l1; j < j_l1+B_L1; j+=SIMD_LEN) {
                                    zmm_cpy = _mm512_load_ps(&_C[idx]);
                                    _mm512_store_ps(&C[i*N+j], zmm_cpy);
                                    idx+=SIMD_LEN;
                                }
                            }
                            
                        }
                    }

                }
            }
            
        }
    }
}