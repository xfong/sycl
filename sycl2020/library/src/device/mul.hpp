// mul kernel

// device side function. This is essentially the function of the kernel
// dst[i] = a[i] * b[i]
template <typename dataT>
void mul_fcn(size_t totalThreads, sycl::nd_item<1> item,
             dataT* dst,
             dataT*  a0,
             dataT*  b0,
             size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
        dst[gid] = a0[gid] * b0[gid];
    }
};

// the function that launches the kernel
template <typename dataT>
void mul_t(size_t blocks, size_t threads, sycl::queue q,
           dataT* dst,
           dataT*  a0,
           dataT*  b0,
           size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(mul_fcn<dataT>, totalThreads, threads,
                             dst,
                              a0,
                              b0,
                               N);
}
