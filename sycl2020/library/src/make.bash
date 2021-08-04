
#!/bin/bash

FN=$1

## Uncomment the following to target CUDA devices
NVTARG=-fsycl-targets=nvptx64-nvidia-cuda-sycldevice
DEBUG=-DNDEBUG
clang++ -shared -fPIC -O3 -fsycl ${NVTARG} ${DEBUG} ${FN}.cpp -o ${FN}.so
