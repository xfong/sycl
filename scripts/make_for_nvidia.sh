#!/bin/bash
dir=$1

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCOMPUTECPP_BITCODE=ptx64 -DComputeCpp_DIR=${ComputeCpp_ROOT_DIR}
