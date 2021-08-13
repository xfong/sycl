// dotproduct kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst = prefactor*(src1x*src2x + src1y*src2y + src1z*src2z)
template<typename dataT>
inline void dotproduct_fcn(sycl::nd_item<3> item,
                           dataT*       dst,
                           dataT  prefactor,
                           dataT*     src1x,
                           dataT*     src1y,
                           dataT*     src1z,
                           dataT*     src2x,
                           dataT*     src2y,
                           dataT*     src2z,
                           size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dataT num1 = src1x[gid];
        dataT num2 = src1y[gid];
        dataT num3 = src1z[gid];
        dataT num4 = src2x[gid];

        num1     *= num4;
        num4      = src2y[gid];
        num2     *= num4;
        num4      = src2z[gid];
        num3     *= num4;
        num4      = num1 + num2 + num3;
        dst[gid]  = prefactor * num4;
    }
}

// the function that launches the kernel
template<typename dataT>
void dotproduct_t(dim3 blocks, dim3 threads, sycl::queue q,
                  dataT*       dst,
                  dataT  prefactor,
                  dataT*     src1x,
                  dataT*     src1y,
                  dataT*     src1z,
                  dataT*     src2x,
                  dataT*     src2y,
                  dataT*     src2z,
                  size_t         N) {
    libMumax3clDeviceFcnCall(dotproduct_fcn<dataT>, blocks, threads,
                                   dst,
                             prefactor,
                                 src1x,
                                 src1y,
                                 src1z,
                                 src2x,
                                 src2y,
                                 src2z,
                                     N);
}
