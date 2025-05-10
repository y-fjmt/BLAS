#include <cuda_runtime.h>
#include "../../include/_utils_cuda.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define ALLCLOSE dallclose
    #define ALLCLOSE_KERNEL dallclose_kernel
#else
    #define data_t float
    #define ALLCLOSE sallclose
    #define ALLCLOSE_KERNEL sallclose_kernel
#endif


namespace cuda {

    __global__ void ALLCLOSE_KERNEL(data_t* input, data_t* other, size_t N, data_t rtol, data_t atol, bool *is_allclose) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        bool is_close = (abs(input[idx]-other[idx]) <= (atol+rtol*abs(other[idx])));
        if (!is_close) {
            *is_allclose = false;
        }
    }


    bool ALLCLOSE(data_t* input, data_t* other, size_t N, data_t rtol=1e-05, data_t atol=1e-08) {
        bool is_allclose = true;
        ALLCLOSE_KERNEL<<<CEIL_DIV(N, 1024), 1024>>>(
            input, other, N, rtol, atol, &is_allclose
        );
        return false;
    }
}
