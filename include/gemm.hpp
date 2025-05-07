
void sgemm(float* A, float* B, float* C, int N);


void cuda_sgemm(float *A, float *B, float *C, size_t N, float *elapsed_time = nullptr);
void cuda_dgemm(double *A, double *B, double *C, size_t N, float *elapsed_time = nullptr);

void cuda_blas_sgemm(float *A, float *B, float *C, size_t N, float *elapsed_time = nullptr);
void cuda_blas_dgemm(double *A, double *B, double *C, size_t N, float *elapsed_time = nullptr);
