NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -g -G -O2
LDFLAGS := -lcublas

all: sgemm_benchmark dgemm_benchmark

sgemm_benchmark: sgemm.o utils.o sbench.o cublas_sgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

dgemm_benchmark: dgemm.o utils.o dbench.o cublas_dgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

sbench.o: ../benchmark/cu_gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

dbench.o: ../benchmark/cu_gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

cublas_sgemm.o: ../cublas/gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

cublas_dgemm.o: ../cublas/gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

sgemm.o: gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

dgemm.o: gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

utils.o: ../utils/utils.cpp
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

clean:
	rm -rf sgemm_benchmark dgemm_benchmark *.o

.PHONY: clean
