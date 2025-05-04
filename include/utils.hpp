#include <iostream>
#include <chrono>
#include <random>


#define CHECK(call)                                   \
{                                                     \
    const cudaError_t error = call;                   \
    if (error != cudaSuccess)                         \
    {                                                 \
       printf("Error: %s:%d,  ", __FILE__, __LINE__); \
       printf("code:%d, reason: %s\n", error,         \
            cudaGetErrorString(error));               \
       exit(1);                                       \
    }                                                 \
}

#define CHECK_CUBLAS(call)                            \
{                                                     \
    const cublasStatus_t status = call;               \
    if (status != CUBLAS_STATUS_SUCCESS)              \
    {                                                 \
       printf("Error: %s:%d,  ", __FILE__, __LINE__); \
       printf("code:%d, reason: cublas error.\n",     \
         status);                                     \
       exit(1);                                       \
    }                                                 \
}


template<typename T>
void init_vector(T *v, size_t N) {
    std::mt19937 engine;
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (int i = 0; i < (int)N; i++) {
        v[i] = dist(engine);
    }
}


template<typename T>
bool allclose(T* C1, T* C2, size_t N, T eps=1e-3) {
    for (int i = 0; i < (int)N; i++) {
        if (abs(C1[i] - C2[i]) > eps) {
            return false;
            break;
        }
    }
    return true;
}

double calc_gflops(std::chrono::high_resolution_clock::time_point start, int N);