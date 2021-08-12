#!/bin/bash

FN=$1
NVTARG="-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"

if [ -f ${FN} ]; then
    rm ${FN}
fi

clang++ -fPIC -O3 -fsycl ${NVTARG} -I/opt/intel/dpcpp/include/sycl -L/opt/intel/dpcpp/lib ${FN}.cpp -lsycl -o ${FN}
