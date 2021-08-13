// regionadds kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
// add region-based scalar to dst:
// dst[i] += LUT[region[i]]
template <typename dataT>
inline void regionadds_fcn(sycl::nd_item<3> item,
                           dataT*       dst,
                           dataT*       LUT,
                           uint8_t* regions,
                           size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        uint8_t r = regions[gid];
        dst[gid] += LUT[r];
    }
}

// the function that launches the kernel
template <typename dataT>
void regionadds_t(dim3 blocks, dim3 threads, sycl::queue q,
                  dataT*       dst,
                  dataT*       LUT,
                  uint8_t* regions,
                  size_t         N) {
    libMumax3clDeviceFcnCall(regionadds_fcn<dataT>, blocks, threads,
                                 dst,
                                 LUT,
                             regions,
                                   N);
}
