#!/bin/bash
dir=$1

cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCOMPUTECPP_BITCODE=spir64 -DCOMPUTECPP_USER_FLAGS=-intelspirmetadata -DComputeCpp_DIR=${ComputeCpp_ROOT_DIR}
