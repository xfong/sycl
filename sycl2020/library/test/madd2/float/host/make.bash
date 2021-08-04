#!/bin/bash

FN=$1

if [ -f ${FN} ]; then
    rm ${FN}
fi

clang++ -fPIC -O3 -fsycl -I../../../../include ${FN}.cpp -o ${FN}
