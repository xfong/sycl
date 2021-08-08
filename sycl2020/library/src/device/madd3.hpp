// madd3 kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// dst = fac1*src1 + fac2*src2 + fac3*src3
template<typename T>
inline void madd3_fcn(size_t totalThreads, sycl::nd_item<1> idx, T* dst, T* src1, T fac1, T* src2, T fac2, T* src3, T fac3, size_t N) {
    size_t myId = idx.get_global_linear_id();
    for (size_t i = myId; i < N; i += totalThreads) {
        T num1 = src1[i];
        num1 *= fac1;
        T num2 = src2[i];
        num2 *= fac2;
        T num3 = src3[i];
        num3 *= fac3;
        num1 += num2 + num3;
        dst[i] = num1;
    }
}

// the function that launches the kernel
template<typename T>
void madd3_t(size_t blocks, size_t threads, sycl::queue q,
             T* dst,
             T* src1,
             T fac1,
             T* src2,
             T fac2,
             T* src3,
             T fac3,
             size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(madd3_fcn<T>, totalThreads, threads,
                             dst,
                             src1,
                             fac1,
                             src2,
                             fac2,
                             src3,
                             fac3,
                             N);
}
