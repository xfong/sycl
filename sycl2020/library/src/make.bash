#!/bin/bash

FN=$1

## Uncomment the following to target CUDA devices
NVTARG="-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
DEBUG=-DNDEBUG

if [ -f ./${FN}.so ]; then
    rm ./${FN}.so
fi
clang++ -shared -fPIC -O3 -fsycl ${NVTARG} ${DEBUG} -I../include ${FN}.cpp -o ${FN}.so

if [ -f ../lib/${FN}.so ]; then
    rm ../lib/${FN}.so
fi

if [ !-d ../lib ]; then
    mkdir ../lib
fi

cp ./${FN}.so ../lib/${FN}.so
