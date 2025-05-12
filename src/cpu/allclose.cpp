#include <cmath>
#include "../../include/_utils.hpp"

#if defined(USE_DOUBLE)
    #define data_t double
    #define ALLCLOSE dallclose
    #define SCALED_ALLCLOSE dscaledAllclose
#else
    #define data_t float
    #define ALLCLOSE sallclose
    #define SCALED_ALLCLOSE sscaledAllclose
#endif

// TODO: implement with vector instruction and omp
bool ALLCLOSE(data_t* input, data_t* other, size_t N, data_t rtol=1e-05, data_t atol=1e-08) {
    bool is_close;
    for (int i = 0; i < N; i++) {
        is_close = (abs(input[i]-other[i]) <= (atol+rtol*abs(other[i])));
        if (!is_close) {
            return false;
        }
    }
    return true;
}

bool SCALED_ALLCLOSE(data_t* input, data_t* other, size_t N, data_t rtol=1e-05, data_t atol=1e-08) {
    data_t factor = std::sqrt(N);
    return ALLCLOSE(input, other, N, factor*rtol, factor*atol);
}
