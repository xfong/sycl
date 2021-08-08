// madd2 kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// dst = fac1*src1 + fac2*src2
template<typename dataT>
inline void madd2_fcn(size_t totalThreads, sycl::nd_item<1> item,
                      dataT* dst,
                      dataT* src1,
                      dataT  fac1,
                      dataT* src2,
                      dataT  fac2,
                      size_t N) {
    for (size_t i = item.get_global_linear_id(); i < N; i += totalThreads) {
        dataT num1 = src1[i];
        dataT num2 = src2[i];
        num1      *= fac1;
        num2      *= fac2;
        num1      += num2;
        dst[i]     = num1;
    }
}

// the function that launches the kernel
template<typename dataT>
void madd2_t(size_t blocks, size_t threads, sycl::queue q,
             dataT* dst,
             dataT* src1,
             dataT  fac1,
             dataT* src2,
             dataT  fac2,
             size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(madd2_fcn<dataT>, totalThreads, threads,
                             dst,
                             src1,
                             fac1,
                             src2,
                             fac2,
                             N);
}
