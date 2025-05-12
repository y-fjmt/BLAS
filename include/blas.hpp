#pragma once

bool sallclose(float* input, float* other, size_t N, float rtol=1e-05, float atol=1e-08);
bool gallclose(double* input, double* other, size_t N, double rtol=1e-05, double atol=1e-08);
bool sscaledAllclose(float* input, float* other, size_t N, float rtol=1e-05, float atol=1e-08);
bool gscaledAllclose(double* input, double* other, size_t N, double rtol=1e-05, double atol=1e-08);
void sgemm(float* A, float* B, float* C, int N);
