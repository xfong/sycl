// regionselect kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
template <typename dataT>
inline void regionselect_fcn(sycl::nd_item<3> item,
                             dataT*       dst,
                             dataT*       src,
                             uint8_t* regions,
                             uint8_t   region,
                             size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dst[gid] = (regions[gid] == region ? src[gid] : (dataT)(0.0));
    }
}

template <typename dataT>
void regionselect_t(dim3 blocks, dim3 threads, sycl::queue q,
                    dataT*       dst,
                    dataT*       src,
                    uint8_t* regions,
                    uint8_t   region,
                    size_t         N) {
    libMumax3clDeviceFcnCall(regionselect_fcn<dataT>, blocks, threads,
                                 dst,
                                 src,
                             regions,
                              region,
                                   N);
}
