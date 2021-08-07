// madd2 kernel

// device side function.This is essentially the function of the kernel
template<typename T>
inline void madd2_fcn(size_t totalThreads, sycl::nd_item<1> idx, T* dst, T* src1, T fac1, T* src2, T fac2, size_t N) {
    size_t myId = idx.get_global_linear_id();
    for (size_t i = myId; i < N; i += totalThreads) {
        T num1 = src1[i];
        T num2 = src2[i];
        num1 *= fac1;
        num2 *= fac2;
        num1 += num2;
        dst[i] = num1;
    }
}

// dst = fac1*src1 + fac2*src2
template<typename T>
void madd2_t(size_t blocks, size_t threads, sycl::queue q,
             T* dst,
             T* src1,
             T fac1,
             T* src2,
             T fac2,
             size_t N) {
    size_t totalThreads = blocks*threads;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(totalThreads), sycl::range<1>(threads)), [=] (sycl::nd_item<1> idx) {
        madd2_fcn<T>(totalThreads, idx, dst, src1, fac1, src2, fac2, N);
    });
}
