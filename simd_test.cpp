#include <iostream>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cblas.h>
#include "../include/utils.hpp"
#include "../include/sgemm.hpp"

#define EPS 1e-3


void my_sgemm_2(float *A, float *B, float *C, int N) {

    // j方向にSIMD命令を適用 (転置, reduceがいらない)

    int simd_len = 16;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j+=simd_len) {
            __m512 zmm_accum = _mm512_setzero_ps();
            for (int k = 0; k < N; k++) {
                __m512 zmm1 = _mm512_set1_ps(A[i*N+k]);
                __m512 zmm2 = _mm512_load_ps(&B[k*N+j]);
                zmm_accum = _mm512_fmadd_ps(zmm1, zmm2, zmm_accum);
            }
            _mm512_store_ps(&C[i*N+j], zmm_accum);
        }
    }
}


int main(int argc, char const *argv[]) {

    int n = 1024;
    float *A, *B, *C1, *C2;
    init_matrix(&A, &B, &C1, &C2, n);

    my_sgemm_2(A, B, C1, n);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, C2, n);

    if (!is_same_mat(C1, C2, n)) {
        std::cout << "result mismatch." << std::endl;
        std::cout << C1[0] << " " << C2[0] << std::endl;
        return 0;
    }

    free_matrix(&A, &B, &C1, &C2);

    return 0;
}


