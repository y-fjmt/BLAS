#include <chrono>

void print_matrix(float* mat, int N);
bool is_same_mat(float* C1, float* C2, int N);
void init_matrix(float** A, float** B, float** C1, float** C2, int N);
void free_matrix(float** A, float** B, float** C1, float** C2);
double calc_gflops(std::chrono::high_resolution_clock::time_point start, int N);