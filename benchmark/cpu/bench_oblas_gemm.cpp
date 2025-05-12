#include <random>
#include <benchmark/benchmark.h>
#include <cblas.h>

#if defined(USE_DOUBLE)
    #define data_t double
    #define OPENBLAS_GEMM_NAME cblas_dgemm
    #define BM_OBLAS_GEMM bench_oblas_dgemm_cpu
    #define BM_NAME "OPENBLAS_DGEMM"
#else
    #define data_t float
    #define OPENBLAS_GEMM_NAME cblas_sgemm
    #define BM_OBLAS_GEMM bench_oblas_sgemm_cpu
    #define BM_NAME "OPENBLAS_SGEMM"
#endif

static void BM_OBLAS_GEMM(benchmark::State& state) {

    // Setup (not mesured)
    int M = state.range(0);
    int N = state.range(1);
    int K = state.range(2);

    std::mt19937 engine;
    std::uniform_real_distribution<data_t> dist(-1.0, 1.0);
    auto generator = [&](){return dist(engine);};

    data_t *A = (data_t*)aligned_alloc(64, M*K*sizeof(data_t));
    data_t *B = (data_t*)aligned_alloc(64, K*N*sizeof(data_t));
    data_t *C = (data_t*)aligned_alloc(64, M*N*sizeof(data_t));

    std::generate(&A[0], &A[M*K], generator);
    std::generate(&B[0], &B[K*N], generator);

    
    // benchmark
    for (auto _ : state) {
        OPENBLAS_GEMM_NAME(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f,
                A, K,
                B, N,
                0.0f,
                C, N);
        benchmark::DoNotOptimize(C);
    }
        

    // Teardown (not mesured)
    double gflop = 2.0 * M * N * K * state.iterations() / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(gflop, benchmark::Counter::kIsRate);

    free(A);
    free(B);
    free(C);
}

// register benchmarks
BENCHMARK(BM_OBLAS_GEMM)->Name(BM_NAME)->Args({1024, 1024, 1024});
BENCHMARK(BM_OBLAS_GEMM)->Name(BM_NAME)->Args({2048, 2048, 2048});
BENCHMARK(BM_OBLAS_GEMM)->Name(BM_NAME)->Args({4096, 4096, 4096});
BENCHMARK(BM_OBLAS_GEMM)->Name(BM_NAME)->Args({8192, 8192, 8192});
BENCHMARK(BM_OBLAS_GEMM)->Name(BM_NAME)->Args({16384, 16384, 16384});
