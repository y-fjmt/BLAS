#include <iostream>
#include "../../include/sgemm.hpp"
#include "../../include/utils.hpp"

#if defined(data_t) && data_t == float
    #define GEMM_NAME cuda_sgemm
    #define CUBLAS_GEMM cuda_blas_sgemm
#elif defined(data_t) && data_t == double
    #define GEMM_NAME cuda_dgemm
    #define CUBLAS_GEMM cuda_blas_dgemm
#else
    #error data_t is not given or invalid data type.
#endif

using namespace std;

int main(int argc, char const *argv[]) {
    
    size_t N = 4096;

    data_t *A = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *B = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *C = (data_t*)malloc(N*N*sizeof(data_t));
    data_t *cublas_C = (data_t*)malloc(N*N*sizeof(data_t));

    init_vector(A, N*N);
    init_vector(B, N*N);

    float kernel_time;

    GEMM_NAME(A, B, C, N, &kernel_time);

    cout << "mygemm: " <<  kernel_time << "(ms)" << endl;

    CUBLAS_GEMM(A, B, cublas_C, N, &kernel_time);

    cout << "cublas: " << kernel_time << "(ms)" << endl;

    cout << (allclose(C, cublas_C, N*N)? "true":"false") << endl;

    cout << C[0] << " " << cublas_C[0] << endl;

    free(A);
    free(B);
    free(C);
    free(cublas_C);
    
    return 0;
}

