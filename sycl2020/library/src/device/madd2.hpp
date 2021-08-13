// madd2 kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst = fac1*src1 + fac2*src2
template<typename dataT>
inline void madd2_fcn(sycl::nd_item<3> item,
                      dataT*  dst,
                      dataT* src1,
                      dataT  fac1,
                      dataT* src2,
                      dataT  fac2,
                      size_t    N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dataT num1 = src1[gid];
        dataT num2 = src2[gid];
        num1      *= fac1;
        num2      *= fac2;
        num1      += num2;
        dst[gid]   = num1;
    }
}

// the function that launches the kernel
template<typename dataT>
void madd2_t(dim3 blocks, dim3 threads, sycl::queue q,
             dataT*  dst,
             dataT* src1,
             dataT  fac1,
             dataT* src2,
             dataT  fac2,
             size_t    N) {
    libMumax3clDeviceFcnCall(madd2_fcn<dataT>, blocks, threads,
                              dst,
                             src1,
                             fac1,
                             src2,
                             fac2,
                                N);
}
