#!/bin/bash

FN=$1

if [ -f ${FN} ]; then
    rm ${FN}
fi

clang++ -fPIC -O3 -I/opt/intel/dpcpp/include/sycl -I../../../include -L../../../lib -L/opt/intel/dpcpp/lib ${FN}.cpp -lmumax3cl_f -lsycl -o ${FN}
