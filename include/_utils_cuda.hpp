#pragma once
#include <curand.h>
#include <cuda_runtime.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    printf("Error code: %d\n", (x)); \
    exit(EXIT_FAILURE);}} while(0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

namespace cuda {
    
}