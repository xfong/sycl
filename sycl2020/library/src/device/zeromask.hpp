// zeromask kernel

#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// set dst to zero in cells where mask != 0
template <typename dataT>
void zeromask_fcn(size_t totalThreads, sycl::nd_item<1> item,
                  dataT* dst,
                  dataT* maskLUT,
                  uint8_t* regions,
                  size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
        if (maskLUT[regions[gid]] != 0) {
            dst[gid] = (dataT)(0.0);
        }
    }
}

// the function that launches the kernel
template <typename dataT>
void zeromask_t(size_t blocks, size_t threads, sycl::queue q,
                 dataT* dst,
                 dataT* maskLUT,
                 uint8_t* regions,
                 size_t N) {
    size_t totalThreads = blocks * threads;
    libMumax3clDeviceFcnCall(zeromask_fcn<dataT>, totalThreads, threads,
                             dst,
                             maskLUT,
                             regions,
                             N);
}
