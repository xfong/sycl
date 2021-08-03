
#!/bin/bash

FN=$1

## Uncomment the following to target CUDA devices
NVTARG=-fsycl-targets=nvptx64-nvidia-cuda-sycldevice
clang++ -fsycl ${NVTARG} ${FN}.cpp -o ${FN}
