// pointwise_div kernel

#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dst = ax / bx
template<typename dataT>
inline void pointwise_div_fcn(size_t totalThreads, sycl::nd_item<1> item,
                              dataT* dst,
                              dataT* ax,
                              dataT* bx,
                              size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
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
void pointwise_div_t(size_t blocks, size_t threads, sycl::queue q,
                     dataT* dst,
                     dataT* ax,
                     dataT* bx,
                     size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(pointwise_div_fcn<dataT>, totalThreads, threads,
                             dst,
                             ax,
                             bx,
                             N);
}
