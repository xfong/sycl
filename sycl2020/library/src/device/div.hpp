// pointwise_div kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst = ax / bx
template<typename dataT>
inline void pointwise_div_fcn(sycl::nd_item<3> item,
                              dataT* dst,
                              dataT*  ax,
                              dataT*  bx,
                              size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dataT c0 = bx[gid];
        dataT c1 = ax[gid];
        if ((c0 == (dataT)(0.0)) || (c1 == (dataT)(0.0))) {
            dst[gid] = (dataT)(0.0);
        } else {
            dst[gid] = c1 / c0;
        }
    }
}

// the function that launches the kernel
template <typename dataT>
void pointwise_div_t(dim3 blocks, dim3 threads, sycl::queue q,
                     dataT* dst,
                     dataT* ax,
                     dataT* bx,
                     size_t N) {
    libMumax3clDeviceFcnCall(pointwise_div_fcn<dataT>, blocks, threads,
                             dst,
                              ax,
                              bx,
                               N);
}
