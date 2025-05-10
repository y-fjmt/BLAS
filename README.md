# BLAS

## Overview
This repository is a BLAS (Basic Linear Algebra Subprograms) 
implementation for studying HPC.

## Build
We use [CMake](https://cmake.org/) to build this library.
You can build it with the following command.
```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make
```
You can build for GPU acceleration with `-DUSE_CUDA=ON`.

## Test
We use [googletest](https://github.com/google/googletest) to check if they compute correct results.
In addition, the tests written in googletest are wrapped by ctest. You can test all of them with `ctest` command.
If you want to test each test individually, use `all_tests`, which is written in googletest.

## Benchmark
comming soon...