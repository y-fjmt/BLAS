CXX := g++
CXXFLAGS := -Wall -O2 -mavx512f -fopenmp -Wl,-rpath=../../OpenBLAS/build/lib
INCLUDES := -I../../OpenBLAS/build/include \
			-I../../googletest/googlemock/include \
			-I../../googletest/googletest/include
LDFLAGS := -lopenblas -lpthread \
		   -L../../OpenBLAS/build/lib \
		   -L../../googletest/build/lib \

all: sgemm_benchmark
# all: sgemm_benchmark sgemm_test

# sgemm_test: sgemm.o sgemm_test.o utils.o
# 	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

sgemm_benchmark: sgemm.o sgemm_bench.o utils.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

sgemm_test.o: ../test/gemm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

sgemm_bench.o: ../benchmark/gemm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

sgemm.o: gemm.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

utils.o: ../utils/utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

clean:
	rm -rf *gemm_benchmark *gemm_test *.o

.PHONY: clean