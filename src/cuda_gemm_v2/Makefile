NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -g -G -O2 -std=c++17
LDFLAGS := -lcublas -lgtest -lgtest_main -lpthread \
		   -L../../googletest/build/lib 
INCLUDES := -I../../googletest/googlemock/include \
			-I../../googletest/googletest/include

all: sgemm_benchmark dgemm_benchmark sgemm_test dgemm_test 

sgemm_test: s_test.o sgemm.o utils.o cublas_sgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

dgemm_test: d_test.o dgemm.o utils.o cublas_dgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

sgemm_benchmark: sgemm.o utils.o sbench.o cublas_sgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

dgemm_benchmark: dgemm.o utils.o dbench.o cublas_dgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

s_test.o: ../test/cuda_gemm.cpp
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@

d_test.o: ../test/cuda_gemm.cpp
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

sbench.o: ../benchmark/cu_gemm.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@

dbench.o: ../benchmark/cu_gemm.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

cublas_sgemm.o: ../cublas/gemm.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@

cublas_dgemm.o: ../cublas/gemm.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

sgemm.o: gemm.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@

dgemm.o: gemm.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@ -DUSE_DOUBLE

utils.o: ../utils/utils.cpp
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -c $^ -o $@

clean:
	rm -rf *gemm_benchmark *gemm_test *.o

.PHONY: clean
