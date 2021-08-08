// dotproduct kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// dst = prefactor*(src1x*src2x + src1y*src2y + src1z*src2z)
template<typename T>
inline void dotproduct_fcn(size_t totalThreads, sycl::nd_item<1> item,
                      T* dst,
                      T  prefactor,
                      T* src1x,
                      T* src1y,
                      T* src1z,
                      T* src2x,
                      T* src2y,
                      T* src2z,
                      size_t N) {
    for (size_t i = item.get_global_linear_id(); i < N; i += totalThreads) {
        T num1 = src1x[i];
        T num2 = src1y[i];
        T num3 = src1z[i];
        T num4 = src2x[i];
        num1 *= num4;
        num4 = src2y[i];
        num2 *= num4;
        num4 = src2z[i];
        num3 *= num4;
        num4 = num1+num2+num3;
        dst[i] = prefactor*num4;
    }
}

// the function that launches the kernel
template<typename T>
void dotproduct_t(size_t blocks, size_t threads, sycl::queue q,
                  T* dst,
                  T  prefactor,
                  T* src1x,
                  T* src1y,
                  T* src1z,
                  T* src2x,
                  T* src2y,
                  T* src2z,
                  size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(dotproduct_fcn<T>, totalThreads, threads,
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
