// madd3 kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst = fac1*src1 + fac2*src2 + fac3*src3
template<typename dataT>
inline void madd3_fcn(sycl::nd_item<3> item,
                      dataT*  dst,
                      dataT* src1,
                      dataT  fac1,
                      dataT* src2,
                      dataT  fac2,
                      dataT* src3,
                      dataT  fac3,
                      size_t    N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dataT num1 = src1[gid];
        num1      *= fac1;
        dataT num2 = src2[gid];
        num2      *= fac2;
        dataT num3 = src3[gid];
        num3      *= fac3;
        num1      += num2 + num3;
        dst[gid]   = num1;
    }
}

// the function that launches the kernel
template<typename dataT>
void madd3_t(dim3 blocks, dim3 threads, sycl::queue q,
             dataT*  dst,
             dataT* src1,
             dataT  fac1,
             dataT* src2,
             dataT  fac2,
             dataT* src3,
             dataT  fac3,
             size_t    N) {
    libMumax3clDeviceFcnCall(madd3_fcn<dataT>, blocks, threads,
                              dst,
                             src1,
                             fac1,
                             src2,
                             fac2,
                             src3,
                             fac3,
                                N);
}
