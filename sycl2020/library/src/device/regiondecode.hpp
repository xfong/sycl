// regiondecode kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
// decode the regions+LUT pair into an uncompressed array
template <typename dataT>
inline void regiondecode_fcn(sycl::nd_item<3> item,
                             dataT*       dst,
                             dataT*       LUT,
                             uint8_t* regions,
                             size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dst[gid] = LUT[regions[gid]];
    }
}

// the function that launches the kernel
template <typename dataT>
void regiondecode_t(dim3 blocks, dim3 threads, sycl::queue q,
                    dataT*       dst,
                    dataT*       LUT,
                    uint8_t* regions,
                    size_t         N) {
    libMumax3clDeviceFcnCall(regiondecode_fcn<dataT>, blocks, threads,
                                 dst,
                                 LUT,
                             regions,
                                   N);
}
