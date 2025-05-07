NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -O2 -std=c++17
LDFLAGS := -lcublas -lgtest -lgtest_main -lpthread \
		   -L../../googletest/build/lib 
INCLUDES := -I../../googletest/googlemock/include \
			-I../../googletest/googletest/include

all: sgemm_benchmark dgemm_benchmark sgemm_test dgemm_test 

sgemm_test: sgemm_test.o sgemm.o utils.o cublas_sgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

dgemm_test: dgemm_test.o dgemm.o utils.o cublas_dgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

sgemm_benchmark: sgemm.o utils.o sbench.o cublas_sgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

dgemm_benchmark: dgemm.o utils.o dbench.o cublas_dgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@

sgemm_test.o: ../test/gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@

dgemm_test.o: ../test/gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@ -DUSE_DOUBLE

sbench.o: ../benchmark/cu_gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@

dbench.o: ../benchmark/cu_gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@ -DUSE_DOUBLE

cublas_sgemm.o: ../cublas/gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@

cublas_dgemm.o: ../cublas/gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@ -DUSE_DOUBLE

sgemm.o: gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@

dgemm.o: gemm.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@ -DUSE_DOUBLE

utils.o: ../utils/utils.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $^ -o $@

clean:
	rm -rf *_benchmark *_test *.o

.PHONY: clean
