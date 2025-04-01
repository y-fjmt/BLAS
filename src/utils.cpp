#include "../include/utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define EPS 1e-3

inline float random_value() {
    return 1.f-2*((float)rand())/RAND_MAX;
}


void print_matrix(float* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.1f", mat[i*N+j]);
        }
        printf("\n");
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

double calc_gflops(std::chrono::high_resolution_clock::time_point start, int N) {
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
    return (2.0 * N * N * N) / (elapsed_sec * 1e9);
}
