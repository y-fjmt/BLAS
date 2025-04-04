#include "../include/sgemm.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cblas.h>
#include <gtest/gtest.h>

using namespace std;

int main(int argc, char const *argv[]) {

    chrono::high_resolution_clock::time_point start, end;
    vector<int> x, y1, y2;

    for (int i = 10; i < 16; i++) {

        int n = (1 << i);
        x.push_back(n);

        float *A = (float*)aligned_alloc(64, n*n*sizeof(float));
        float *B = (float*)aligned_alloc(64, n*n*sizeof(float));
        float *C1 = (float*)aligned_alloc(64, n*n*sizeof(float));
        float *C2 = (float*)aligned_alloc(64, n*n*sizeof(float));
        init_matrix(&A, &B, &C1, &C2, n);

        start = chrono::system_clock::now();
        my_sgemm(A, B, C1, n);
        y1.push_back(calc_gflops(start, n));

        start = chrono::system_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 0, C2, n);
        y2.push_back(calc_gflops(start, n));

        if (!is_same_mat(C1, C2, n)) {
            cout << "result mismatch." << endl;
            return 0;
        }
        int last_idx = x.size()-1;
        cout << "n=" << x[last_idx] << ", " << y1[last_idx] << "/" << y2[last_idx] << endl;

        free_matrix(&A, &B, &C1, &C2);
    }

    std::ofstream file("plot/flops.txt");
    if (file.is_open()) {
        file << "# x y1 y2" << endl;
        for (int i = 0; i < (int)x.size(); i++) {
            file << x[i] << ", " << y1[i] << ", " << y2[i] << endl;
        }
        file.close();
    }
    
    return 0;
}
