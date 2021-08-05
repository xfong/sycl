// madd3 kernel

// dst = fac1*src1 + fac2*src2 + fac3*src3
template<typename T>
void madd3_t(size_t blocks, size_t threads, sycl::queue q,
             T* dst,
             T fac1, T* src1,
             T fac2, T* src2,
             T fac3, T* src3,
             size_t N) {
    size_t totalThreads = blocks*threads;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(totalThreads), sycl::range<1>(threads)), [=] (sycl::nd_item<1> idx) {
        size_t myId = idx.get_global_linear_id();
        for (size_t i = myId; i < N; i += totalThreads) {
            T num1 = src1[i];
            T num2 = src2[i];
            T num3 = src3[i];
            dst[i] = fac1*num1 + fac2*num2 + fac3*num3;
        }
    });
}
