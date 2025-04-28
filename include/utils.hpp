#include <chrono>
#include <random>

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