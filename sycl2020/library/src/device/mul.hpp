// mul kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst[i] = a[i] * b[i]
template <typename dataT>
inline void mul_fcn(sycl::nd_item<3> item,
                    dataT* dst,
                    dataT*  a0,
                    dataT*  b0,
                    size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dst[gid] = a0[gid] * b0[gid];
    }
};

// the function that launches the kernel
template <typename dataT>
void mul_t(dim3 blocks, dim3 threads, sycl::queue q,
           dataT* dst,
           dataT*  a0,
           dataT*  b0,
           size_t   N) {
    libMumax3clDeviceFcnCall(mul_fcn<dataT>, blocks, threads,
                             dst,
                              a0,
                              b0,
                               N);
}
