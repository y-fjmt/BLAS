#include <iostream>
#include <utility>
#include <chrono>
#include <cblas.h>
#include "../../include/gemm.hpp"
#include "../../include/utils.hpp"

using namespace std;

#define data_t float
#define GEMM_NAME sgemm
#define OPENBLAS_GEMM_NAME cblas_sgemm

int main(int argc, char const *argv[]) {
    
    size_t N = 4096;
    int n_warmup  = 1;
    int n_repeats = 5;

    cout << "CPU: " << cpu_name() << endl;
    cout << "Precision: " << TYPE_NAME(data_t) << endl;

    data_t *A = (data_t*)aligned_alloc(64, N*N*sizeof(data_t));
    data_t *B = (data_t*)aligned_alloc(64, N*N*sizeof(data_t));
    data_t *C = (data_t*)aligned_alloc(64, N*N*sizeof(data_t));
    data_t *OpenBLAS_C = (data_t*)aligned_alloc(64, N*N*sizeof(data_t));

    init_vector(A, N*N);
    init_vector(B, N*N);

    float diff;
    chrono::system_clock::time_point st, et;

    // GEMM
    cout << "[GEMM]" << endl;
    for (int i = 0; i < n_warmup; i++) {
        GEMM_NAME(A, B, C, N);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        GEMM_NAME(A, B, C, N);
        et = chrono::system_clock::now();
        diff = chrono::duration<double, std::milli>(et-st).count();
        printf("%.2f(ms), %.2f(GFLOPS)\n", diff,  _mm_gflops(diff/1000, N));
    }

    // OpenBLAS GEMM
    cout << "[OpenBLAS]" << endl;
    for (int i = 0; i < n_warmup; i++) {
        OPENBLAS_GEMM_NAME(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                            N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        OPENBLAS_GEMM_NAME(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                            N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
        et = chrono::system_clock::now();
        diff = chrono::duration<double, std::milli>(et-st).count();
        printf("%.2f(ms), %.2f(GFLOPS)\n", diff,  _mm_gflops(diff/1000, N));
    }

    cout << "allclose: " << (allclose(C, OpenBLAS_C, N*N)? "true":"false") << endl;

    free(A);
    free(B);
    free(C);
    free(OpenBLAS_C);
    
    return 0;
}

