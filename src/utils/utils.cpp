#include <iostream>
#include <cstdlib>
#include <chrono>


double calc_gflops(std::chrono::high_resolution_clock::time_point start, int N) {
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
    return (2.0 * N * N * N) / (elapsed_sec * 1e9);
}
