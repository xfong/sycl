// minimize kernel

// device side function. This is essentially the function of the kernel
// Steepest descent energy minimizer
template <typename dataT>
void minimize_fcn(size_t totalThreads, sycl::nd_item<1> item,
                  dataT*  mx_, dataT*  my_, dataT*  mz_,
                  dataT* m0x_, dataT* m0y_, dataT* m0z_,
                  dataT*  tx_, dataT*  ty_, dataT*  tz_,
                  dataT dt, size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
        sycl::vec<dataT, 3> m0 = {m0x_[gid], m0y_[gid], m0z_[gid]};
        sycl::vec<dataT, 3> t = {tx_[gid], ty_[gid], tz_[gid]};

        sycl::vec<dataT, 3> t2 = dt*dt*sycl::dot(t, t);
        sycl::vec<dataT, 3> result = ((dataT)(4.0) - t2) * m0 + (dataT)(4.0) * dt * t;
        sycl::vec<dataT, 3> divisor = (dataT)(4.0) + t2;

        mx_[gid] = result.x() / divisor;
        my_[gid] = result.y() / divisor;
        mz_[gid] = result.z() / divisor;
    }
}

// the function that launches the kernel
template <typename dataT>
void minimize_t(size_t blocks, size_t threads, sycl::queue q,
                  dataT*  mx_, dataT*  my_, dataT*  mz_,
                  dataT* m0x_, dataT* m0y_, dataT* m0z_,
                  dataT*  tx_, dataT*  ty_, dataT*  tz_,
                  dataT dt, size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(mul_fcn<dataT>, totalThreads, threads,
                              mx_,  my_,  mz_,
                             m0x_, m0y_, m0z_,
                              tx_,  ty_,  tz_,
                               dt,    N);
}
