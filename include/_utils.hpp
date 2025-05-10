#pragma once
#include <iostream>
#include <chrono>
#include <random>
#include <string>

#define TO_STRING(x) #x
#define TYPE_NAME(x) TO_STRING(x)

using namespace std;

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



double _mm_gflops(double sec, int N);
string cpu_name();

template<typename T>
void init_vector(T *v, size_t N) {
    std::mt19937 engine;
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (int i = 0; i < (int)N; i++) {
        v[i] = dist(engine);
    }
}