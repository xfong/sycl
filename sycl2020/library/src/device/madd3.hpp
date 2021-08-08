// madd3 kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// dst = fac1*src1 + fac2*src2 + fac3*src3
template<typename dataT>
inline void madd3_fcn(size_t totalThreads, sycl::nd_item<1> item,
                      dataT* dst,
                      dataT* src1,
                      dataT  fac1,
                      dataT* src2,
                      dataT  fac2,
                      dataT* src3,
                      dataT  fac3,
                      size_t N) {
    for (size_t i = item.get_global_linear_id(); i < N; i += totalThreads) {
        dataT num1 = src1[i];
        num1      *= fac1;
        dataT num2 = src2[i];
        num2      *= fac2;
        dataT num3 = src3[i];
        num3      *= fac3;
        num1      += num2 + num3;
        dst[i]     = num1;
    }
}

// the function that launches the kernel
template<typename dataT>
void madd3_t(size_t blocks, size_t threads, sycl::queue q,
             dataT* dst,
             dataT* src1,
             dataT  fac1,
             dataT* src2,
             dataT  fac2,
             dataT* src3,
             dataT  fac3,
             size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(madd3_fcn<dataT>, totalThreads, threads,
                             dst,
                             src1,
                             fac1,
                             src2,
                             fac2,
                             src3,
                             fac3,
                             N);
}
