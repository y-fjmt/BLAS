CXX := /usr/bin/g++
CXXFLAGS := -g -Wall -O2 -fopenmp
LDFLAGS := -lcublas -lcudart -lstdc++ \
		   -Lgoogletest/build/lib
INCLUDES := -IOpenBLAS/build/include \
			-Igoogletest/googlemock/include \
			-Igoogletest/googletest/include
FLAGS := -fopenmp

NVCC := /usr/local/cuda/bin/nvcc
NVCCFLAGS := -g -G \
			-Xcompiler -fopenmp \
			--generate-code \
			arch=compute_90,code=sm_90
SRC := src
DST := build

$(DST):
	mkdir -p $(DST)


$(DST)/utils.o: $(SRC)/utils/utils.cpp | $(DST)
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)


test_cu_sgemm.out: $(DST)/utils.o
	$(NVCC) $(NVCCFLAGS) -Ddata_t=float \
		-c $(SRC)/gemm_cublas.cu -o $(DST)/cu_sgemm_blas.o
	$(NVCC) $(NVCCFLAGS) -Ddata_t=float \
		-c $(SRC)/gemm.cu -o $(DST)/cu_sgemm.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) -Ddata_t=float\
		-c $(SRC)/test/cu_gemm.cpp -o $(DST)/test_cu_sgemm.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) \
		$(DST)/cu_sgemm.o $(DST)/cu_sgemm_blas.o $(DST)/test_cu_sgemm.o -o $@

clean:
	rm -rf $(DST) *.out *.o

.PHONY: clean
