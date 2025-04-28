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

pair<double, double> mean_std(vector<double> v) {
    double mean = reduce(v.begin(), v.end()) / v.size();
    double std = 0.0;
    for (double x: v) {
        std += pow(mean-x ,2);
    }
    std = sqrt(std / v.size());
    return make_pair(mean, std);    
}

int main(int argc, char const *argv[]) {
    
    size_t N = 64;
    int n_warmup  = 1;
    int n_repeats = 1;

    data_t *A = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *B = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *C = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *OpenBLAS_C = (data_t*)malloc(N*N*sizeof(data_t));

    init_vector(A, N*N);
    init_vector(B, N*N);

    float kernel_time;
    chrono::system_clock::time_point st, et;
    vector<double> openblas_elapsed;
    vector<double> cuda_elapsed, cuda_kernel_elapsed;

    // CUDA GEMM
    for (int i = 0; i < n_warmup; i++) {
        GEMM_NAME(A, B, C, N, &kernel_time);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        GEMM_NAME(A, B, C, N, &kernel_time);
        et = chrono::system_clock::now();
        cuda_kernel_elapsed.push_back(kernel_time);
        cuda_elapsed.push_back(
            chrono::duration_cast<chrono::milliseconds>(et-st).count()
        );
    }

    // OpenBLAS GEMM
    for (int i = 0; i < n_warmup; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                    N, N, N, 1, A, N, B, N, 0, OpenBLAS_C, N);
        et = chrono::system_clock::now();
        openblas_elapsed.push_back(
            chrono::duration_cast<chrono::milliseconds>(et-st).count()
        );
    }


    cout << "allclose: " << (allclose(C, OpenBLAS_C, N*N)? "true":"false") << endl;

    free(A);
    free(B);
    free(C);
    free(OpenBLAS_C);
    
    return 0;
}

