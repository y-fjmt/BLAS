#include <immintrin.h>

#define BLOCK 128
// cache 80K/2M/105M
#define B_L1 32
#define B_L2 256
#define B_L3 1024


void print_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.1f", mat[i*N+j]);
            if ((j+1)%2 == 0) printf(" |");
        }
        printf("\n");
        if ((i+1)%2 == 0){
            for (int k = 0; k < N; k++){
                printf("------");
            }
            printf("\n");
        }
    }
}


bool is_same_mat(float* C1, float* C2, int N) {
    for (int i = 0; i < N*N; i++) {
        if (abs(C1[i] - C2[i]) > EPS) {
            return false;
            break;
        }
    }
    return true;
}


inline int idx(int N, int i, int j) {

    int N_L3 = N / B_L3;
    int N_L2 = B_L3 / B_L2;
    int N_L1 = B_L2 / B_L1;
    int N_L0 = B_L1 / 1;

    int i_l3 = i / B_L3, j_l3 = j / B_L3;
    int i_l2 = (i%B_L3)/B_L2, j_l2 = (j%B_L3) / B_L2;
    int i_l1 = (i%B_L2)/B_L1, j_l1 = (j%B_L2) / B_L1;
    int i_l0 = i%B_L1, j_l0 = j%B_L1;

    return (i_l3 * N_L3 + j_l3) * (B_L3 * B_L3) + 
           (i_l2 * N_L2 + j_l2) * (B_L2 * B_L2) + 
           (i_l1 * N_L1 + j_l1) * (B_L1 * B_L1) + 
           (i_l0 * B_L1 + j_l0);
}


void _my_sgemm_comp(float* A, float* B, float* C, int N, int I, int J, int K) {
    __m512 zmm1, zmm2, zmm_accum;
    for (int i = I; i < I+B_L1; i++) {
        for (int j = J; j < J+B_L1; j++) {
            zmm_accum = _mm512_setzero_ps();
            for (int k = K; k < K+B_L1; k+=16) {
                zmm1 = _mm512_load_ps(&A[idx(N, i, k)]);
                zmm2 = _mm512_load_ps(&B[idx(N, j, k)]);
                zmm_accum = _mm512_fmadd_ps(zmm1, zmm2, zmm_accum);
            }
            C[idx(N, i, j)] += _mm512_reduce_add_ps(zmm_accum);
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

void _reorder(float* A, float* B, float* C, float* _A, float* _B, float* _C, int N) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int dst_idx = idx(N, i, j);
            _A[dst_idx] = A[i*N + j];
            _B[dst_idx] = B[j*N + i];
            _C[dst_idx] = C[i*N + j];
        }
    }
}

void _order_C(float* C, float* _C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            _C[i*N+j] = C[idx(N, i, j)];
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
    // - 6. [x] ループアンローリング (34.3625[GFLOPS])
    // - 7. [x] OpenMPで並列化 (156.067[GFLOPS])
    // - 8. [x] ブロッキングレベルを3段階に変更 (少し遅くなった)
    // - 9. [x] ブロッキングしたときに連続アクセスできるように並び替えた
    //      (アドレス変換の実装が微妙でかなり遅くなった)

    float *_A, *_B, *_C;
    posix_memalign((void**)&_A, 64, N*N*sizeof(float));
    posix_memalign((void**)&_B, 64, N*N*sizeof(float));
    posix_memalign((void**)&_C, 64, N*N*sizeof(float));

    _reorder(A, B, C, _A, _B, _C, N);

    _my_sgemm_l3(_A, _B, _C, N);

    _order_C(_C, C, N);

    free(_A);
    free(_B);
    free(_C);
}


void init_matrix(float** A, float** B, float** C1, float** C2, int N) {
    posix_memalign((void**)A, 64, N*N*sizeof(float));
    posix_memalign((void**)B, 64, N*N*sizeof(float));
    posix_memalign((void**)C1, 64, N*N*sizeof(float));
    posix_memalign((void**)C2, 64, N*N*sizeof(float));
    for (int i = 0; i < N*N; i++) {
        (*A)[i] = random_value();
        (*B)[i] = random_value();
        (*C1)[i] = 0.f;
        (*C2)[i] = 0.f;
    }
}

void free_matrix(float** A, float** B, float** C1, float** C2) {
    free(*A);
    free(*B);
    free(*C1);
    free(*C2);
}

double calc_gflops(chrono::high_resolution_clock::time_point start, int N) {
    auto end = chrono::high_resolution_clock::now();
    double elapsed_sec = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e6;
    return (2.0 * N * N * N) / (elapsed_sec * 1e9);
}


int main(int argc, char const *argv[]) {

    double flops;
    chrono::high_resolution_clock::time_point start, end;
    vector<int> x, y1, y2;

    for (int i = 10; i < 16; i++) {

        int n = (1 << i);
        x.push_back(n);

        float *A, *B, *C1, *C2;
        init_matrix(&A, &B, &C1, &C2, n);

        start = chrono::system_clock::now();
        my_sgemm(A, B, C1, n);
        y1.push_back(calc_gflops(start, n));

        start = chrono::system_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, C2, n);
        y2.push_back(calc_gflops(start, n));

        if (!is_same_mat(C1, C2, n)) {
            cout << "result mismatch." << endl;
            return 0;
        }
        int last_idx = x.size()-1;
        cout << "n=" << x[last_idx] << ", " << y1[last_idx] << "/" << y2[last_idx] << endl;

        free_matrix(&A, &B, &C1, &C2);
    }

    std::ofstream file("plot/flops.txt");
    if (file.is_open()) {
        file << "# x y1 y2" << endl;
        for (int i = 0; i < (int)x.size(); i++) {
            file << x[i] << ", " << y1[i] << ", " << y2[i] << endl;
        }
        file.close();
    }
    
    return 0;
}
