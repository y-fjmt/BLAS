#include <omp.h>
#include <immintrin.h>

#include "../include/utils.hpp"
#include <cstdio>

#define B_L1 64
#define B_L2 256
#define B_L3 512
#define SIMD_LEN 16

// Note:
// - B はブロック単位で見ると転置になるがL1ブロックの中は非転置

void _order(float* A, float* B, float* C, float* _A, float* _B, float* _C, int N) {

    __m512 zmm_cpy, zmm_zeros;
    zmm_zeros = _mm512_setzero_ps();

    #pragma omp parallel for collapse(2)
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
                            // Bはブロック単位で転置されている
                            // L1ブロック内はSIMD_LENを単位として転置した方が良い気がする
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


void reorder(float* C, float *_C, int N) {

    __m512 zmm_cpy;

    #pragma omp parallel for collapse(2)
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


void my_sgemm_orderd(float* A, float* B, float* C, int N) {

    float *_A = (float*)aligned_alloc(64, N*N*sizeof(float));
    float *_B = (float*)aligned_alloc(64, N*N*sizeof(float));
    float *_C = (float*)aligned_alloc(64, N*N*sizeof(float));

    int N_L3 =    N / B_L3, B_L3_2 = B_L3 * B_L3;
    int N_L2 = B_L3 / B_L2, B_L2_2 = B_L2 * B_L2;
    int N_L1 = B_L2 / B_L1, B_L1_2 = B_L1 * B_L1;

    _order(A, B, C, _A, _B, _C, N);


    // TODO:
    // - [ ] index計算をポインタに書き換える
    // L3 Blocks
    #pragma omp parallel for collapse(2)
    for (int i_l3 = 0; i_l3 < N_L3; i_l3++) {
        for (int j_l3 = 0; j_l3 < N_L3; j_l3++) {
            for (int k_l3 = 0; k_l3 < N_L3; k_l3++) {
                
                int l3_idx_a = ((i_l3 * N_L3) + k_l3) * B_L3_2;
                int l3_idx_b = ((j_l3 * N_L3) + k_l3) * B_L3_2;
                int l3_idx_c = ((i_l3 * N_L3) + j_l3) * B_L3_2;

                // L2 Blocks
                for (int i_l2 = 0; i_l2 < N_L2; i_l2++) {
                    for (int j_l2 = 0; j_l2 < N_L2; j_l2++) {
                        for (int k_l2 = 0; k_l2 < N_L2; k_l2++) {

                            int l2_idx_a = l3_idx_a + ((i_l2 * N_L2) + k_l2) * B_L2_2;
                            int l2_idx_b = l3_idx_b + ((j_l2 * N_L2) + k_l2) * B_L2_2;
                            int l2_idx_c = l3_idx_c + ((i_l2 * N_L2) + j_l2) * B_L2_2;

                            // L1 Blocks
                            for (int i_l1 = 0; i_l1 < N_L1; i_l1++) {
                                for (int j_l1 = 0; j_l1 < N_L1; j_l1++) {
                                    for (int k_l1 = 0; k_l1 < N_L1; k_l1++) {
                                        
                                        int l1_idx_a = l2_idx_a + ((i_l1 * N_L1) + k_l1) * B_L1_2;
                                        int l1_idx_b = l2_idx_b + ((j_l1 * N_L1) + k_l1) * B_L1_2;
                                        int l1_idx_c = l2_idx_c + ((i_l1 * N_L1) + j_l1) * B_L1_2;

                                        __m512 zmm1, zmm2, zmm_accum;
                                        
                                        // computation
                                        for (int i = 0; i < B_L1; i++) {
                                            for (int j = 0; j < B_L1; j+=SIMD_LEN) {

                                                zmm_accum = _mm512_load_ps(&_C[l1_idx_c+i*B_L1+j]);
                                                for (int k = 0; k < B_L1; k++) {

                                                    zmm1 = _mm512_set1_ps(_A[l1_idx_a+i*B_L1+k]);
                                                    zmm2 = _mm512_load_ps(&_B[l1_idx_b+k*B_L1+j]);
                                                    zmm_accum = _mm512_fmadd_ps(zmm1, zmm2, zmm_accum);

                                                }

                                                _mm512_store_ps(&_C[l1_idx_c+i*B_L1+j], zmm_accum);
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

    reorder(C, _C, N);

    free(_A);
    free(_B);
    free(_C);
}