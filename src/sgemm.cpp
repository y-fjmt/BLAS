#include <omp.h>
#include <immintrin.h>

#define B_L1 64
#define B_L2 256
#define B_L3 1024


void _my_sgemm_comp(float* A, float* B, float* C, int N, int I, int J, int K) {
    __m512 zmm1, zmm2, zmm_accum;
    for (int i = I; i < I+B_L1; i++) {
        for (int j = J; j < J+B_L1; j++) {

            zmm_accum = _mm512_setzero_ps();

        #define CALC(X) \
            zmm1 = _mm512_load_ps(&A[i*N+K+(X)*16]); \
            zmm2 = _mm512_load_ps(&B[j*N+K+(X)*16]); \
            zmm_accum = _mm512_fmadd_ps(zmm1, zmm2, zmm_accum);

            CALC(0)
            CALC(1)
            CALC(2)
            CALC(3)

            C[i*N+j] += _mm512_reduce_add_ps(zmm_accum);
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

    // - 0. [x] ナイーブな実装 (0.404791[GFLOPS])
    // - 1. [x] Bを転置して空間局所性を高める (1.9169[GFLOPS])
    // - 2. [x] AVX512でベクトル化 (14.3965[GFLOPS])
    // - 3. [x] 64B境界でアライメント (変化なし)
    // - 4. [x] B=64でブロッキング (17.2415[GFLOPS])
    // - 5. [x] ベクトル命令の変更 mul+add -> fmadd (27.3119[GFLOPS])
    // - 6. [x] ブロッキングレベルを3段階に変更 (少し遅くなった)

    // - 6. [x] ループアンローリング (34.3625[GFLOPS])
    // - 7. [x] OpenMPで並列化(n_thread=32) (156.067[GFLOPS])

    float *_B;
    posix_memalign((void**)&_B, 64, N*N*sizeof(float));

    # pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            _B[i*N+j] = B[j*N+i];
            C[i*N+j] = 0.f;
        }
    }

    _my_sgemm_l3(A, _B, C, N);

    free(_B);
}