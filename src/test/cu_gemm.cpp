#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <cblas.h>
#include "../../include/sgemm.hpp"
#include "../../include/utils.hpp"

#define data_t float

#if defined(data_t) && data_t == float
    #define GEMM_NAME cuda_sgemm
#elif defined(data_t) && data_t == double
    #define GEMM_NAME cuda_dgemm
#else
    #error data_t is not given or invalid data type.
#endif

using namespace std;

double _mm_gflops(double sec, int N) {
    return (2*N*N) / sec / 1000 / 1000;
}

int main(int argc, char const *argv[]) {
    
    size_t N = 1024;
    int n_warmup  = 1;
    int n_repeats = 5;

    data_t *A = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *B = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *C = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *OpenBLAS_C = (data_t*)malloc(N*N*sizeof(data_t));

    init_vector(A, N*N);
    init_vector(B, N*N);

    float diff, kernel_time;
    chrono::system_clock::time_point st, et;
    vector<double> openblas_elapsed;
    vector<double> cuda_elapsed, cuda_kernel_elapsed;

    // CUDA GEMM
    cout << "[CUDA]" << endl;
    for (int i = 0; i < n_warmup; i++) {
        GEMM_NAME(A, B, C, N, &kernel_time);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        GEMM_NAME(A, B, C, N, &kernel_time);
        et = chrono::system_clock::now();
        diff = chrono::duration<double, std::milli>(et-st).count();
        printf("%.2f(ms), %.2f(GFLOPS) / %.2f(ms), %.2f(GFLOPS)\n", 
            diff,  _mm_gflops(diff/1000, N), kernel_time, _mm_gflops(kernel_time/1000, N));
    }

    // OpenBLAS GEMM
    cout << "[OpenBLAS]" << endl;
    for (int i = 0; i < n_warmup; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
        et = chrono::system_clock::now();
        diff = chrono::duration<double, std::milli>(et-st).count();
        cout << diff << "(ms), " << _mm_gflops(diff/1000, N) << "(GFLOPS)" << endl;
    }


    cout << "allclose: " << (allclose(C, OpenBLAS_C, N*N)? "true":"false") << endl;

    free(A);
    free(B);
    free(C);
    free(OpenBLAS_C);
    
    return 0;
}

