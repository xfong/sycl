// zeromask kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// set dst to zero in cells where mask != 0
template <typename dataT>
inline void zeromask_fcn(sycl::nd_item<3> item,
                         dataT*       dst,
                         dataT*   maskLUT,
                         uint8_t* regions,
                         size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        if (maskLUT[regions[gid]] != 0) {
            dst[gid] = (dataT)(0.0);
        }
    }
}

// the function that launches the kernel
template <typename dataT>
void zeromask_t(dim3 blocks, dim3 threads, sycl::queue q,
                 dataT*       dst,
                 dataT*   maskLUT,
                 uint8_t* regions,
                 size_t         N) {
    libMumax3clDeviceFcnCall(zeromask_fcn<dataT>, blocks, threads,
                                 dst,
                             maskLUT,
                             regions,
                                   N);
}
