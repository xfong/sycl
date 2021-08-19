#!/bin/bash

FN=$1

## Uncomment the following to target CUDA devices
NVTARG="-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
DEBUG=-DNDEBUG

if [ -f ./libmumax3cl_f.so ]; then
    rm ./libmumax3cl_f.so
fi
if [ -f ./libmumax3cl_d.so ]; then
    rm ./libmumax3cl_d.so
fi
clang++ -shared -fPIC -O3 -fsycl ${NVTARG} ${DEBUG} libmumax3cl_t.cpp -o libmumax3cl_f.so
clang++ -shared -fPIC -O3 -fsycl ${NVTARG} ${DEBUG} -D__REAL_IS_DOUBLE__ libmumax3cl_t.cpp -o libmumax3cl_d.so

if [ ! -d ../lib ]; then
    mkdir ../lib
else
    if [ -f ../lib/libmumax3cl_f.so ]; then
        rm ../lib/libmumax3cl_f.so
    fi
    if [ -f ../lib/libmumax3cl_d.so ]; then
        rm ../lib/libmumax3cl_d.so
    fi
fi

cp ./libmumax3cl_f.so ../lib/libmumax3cl_f.so
cp ./libmumax3cl_d.so ../lib/libmumax3cl_d.so

