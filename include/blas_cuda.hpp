#pragma once

namespace cuda {
    bool sallclose(float* input, float* other, size_t N, float rtol=1e-05, float atol=1e-08);
    bool dallclose(double* input, double* other, size_t N, double rtol=1e-05, double atol=1e-08);
    void sgemm(float *A, float *B, float *C, size_t N);
    void dgemm(double *A, double *B, double *C, size_t N);
}