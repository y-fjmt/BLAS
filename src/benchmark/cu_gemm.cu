#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include "../../include/sgemm.hpp"
#include "../../include/utils.hpp"
#include "../../include/cublas_wrapper.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_NAME cuda_dgemm
    #define CUBLAS_GEMM_NAME cublas_dgemm
#else
    #define data_t float
    #define GEMM_NAME cuda_sgemm
    #define CUBLAS_GEMM_NAME cublas_sgemm
#endif

#define TO_STRING(x) #x
#define TYPE_NAME(x) TO_STRING(x)

using namespace std;

double _mm_gflops(double sec, int N) {
    return (2.0*N*N*N) / (sec * 1e9);
}


int main(int argc, char const *argv[]) {
    
    size_t N = 4096;
    int n_warmup  = 1;
    int n_repeats = 5;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "GPU: " << prop.name << endl;
    cout << "Precision: " << TYPE_NAME(data_t) << endl;

    data_t *A = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *B = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *C = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *cuBLAS_C = (data_t*)malloc(N*N*sizeof(data_t));

    init_vector(A, N*N);
    init_vector(B, N*N);

    float diff, kernel_time;
    chrono::system_clock::time_point st, et;

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
    cout << "[cuBLAS]" << endl;
    for (int i = 0; i < n_warmup; i++) {
        CUBLAS_GEMM_NAME(A, B, cuBLAS_C, N, &kernel_time);
    }
    for (int i = 0; i < n_repeats; i++) {
        st = chrono::system_clock::now();
        CUBLAS_GEMM_NAME(A, B, cuBLAS_C, N, &kernel_time);
        et = chrono::system_clock::now();
        diff = chrono::duration<double, std::milli>(et-st).count();
        printf("%.2f(ms), %.2f(GFLOPS) / %.2f(ms), %.2f(GFLOPS)\n", 
            diff,  _mm_gflops(diff/1000, N), kernel_time, _mm_gflops(kernel_time/1000, N));
    }

    cout << "allclose: " << (allclose(C, cuBLAS_C, N*N)? "true":"false") << endl;

    free(A);
    free(B);
    free(C);
    free(cuBLAS_C);
    
    return 0;
}

