// dotproduct kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst = sqrt(dot(A,A))
template <typename dataT>
inline void vecnorm_fcn(sycl::nd_item<3> item,
                    dataT* dst,
                    dataT*  a0, dataT* a1, dataT* a2,
                    size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        sycl::vec<dataT, 3> A = {a0[gid], a1[gid], a2[gid]};
        dst[gid] = sycl::sqrt(sycl::dot(A, A));
    }
}

// the function that launches the kernel
template <typename dataT>
void vecnorm_t(dim3 blocks, dim3 threads, sycl::queue q,
               dataT* dst,
               dataT*  a0, dataT* a1, dataT* a2,
               size_t   N) {
    libMumax3clDeviceFcnCall(vecnorm_fcn<dataT>, blocks, threads,
                             dst,
                              a0, a1, a2,
                               N);
}
