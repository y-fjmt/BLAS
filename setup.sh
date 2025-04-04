#!/bin/bash

# Build OpenBLAS
cd OpenBLAS
make -j
make PREFIX=$(pwd)/build install
cd ..


# Build GoogleTest
cd googletest
mkdir -p build
cd build
cmake ..
make -j
cd ../..