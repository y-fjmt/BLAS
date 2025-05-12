#include <cublas_v2.h>
#include <curand.h>
#include <benchmark/benchmark.h>
#include "../../include/blas_cuda.hpp"
#include "../../include/_utils_cuda.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define GEMM_CUBLAS cublasDgemm
    #define RANDOM curandGenerateNormalDouble
    #define BM_GEMM bench_dgemm_cublas
    #define BM_NAME "CUBLAS_DGEMM"
#else
    #define data_t float
    #define GEMM_CUBLAS cublasSgemm
    #define RANDOM curandGenerateNormal
    #define BM_GEMM bench_sgemm_cublas
    #define BM_NAME "CUBLAS_SGEMM"
#endif

static void BM_GEMM(benchmark::State& state) {
    
    // Setup (not mesured)
    int M = state.range(0);
    int N = state.range(1);
    int K = state.range(2);

    curandGenerator_t gen;

    data_t *A, *B, *C;
    CUDA_CALL(cudaMalloc((void**)&A, N*K*sizeof(data_t)));
    CUDA_CALL(cudaMalloc((void**)&B, K*M*sizeof(data_t)));
    CUDA_CALL(cudaMalloc((void**)&C, N*M*sizeof(data_t)));

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CALL(RANDOM(gen, A, N*K, 0.0, 1.0));
    CURAND_CALL(RANDOM(gen, B, K*M, 0.0, 1.0));

    data_t alpha = 1.0f;
    data_t beta = 0.0f;
    cublasHandle_t handle; 
    cublasCreate(&handle);

    // benchmark
    for (auto _ : state) {
        GEMM_CUBLAS(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            B, N,
            A, N,
            &beta,
            C, N);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(C);
    }

    // Teardown (not mesured)
    double gflop = 2.0 * M * N * K * state.iterations() / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(gflop, benchmark::Counter::kIsRate);

    CURAND_CALL(curandDestroyGenerator(gen));
    cublasDestroy(handle);
    CUDA_CALL(cudaFree(A));
    CUDA_CALL(cudaFree(B));
    CUDA_CALL(cudaFree(C));
}

// register benchmarks
BENCHMARK(BM_GEMM)->Name(BM_NAME)->Args({1024, 1024, 1024});
BENCHMARK(BM_GEMM)->Name(BM_NAME)->Args({2048, 2048, 2048});
BENCHMARK(BM_GEMM)->Name(BM_NAME)->Args({4096, 4096, 4096});
BENCHMARK(BM_GEMM)->Name(BM_NAME)->Args({8192, 8192, 8192});
BENCHMARK(BM_GEMM)->Name(BM_NAME)->Args({16384, 16384, 16384});
