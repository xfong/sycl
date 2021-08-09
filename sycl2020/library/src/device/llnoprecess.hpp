// llnoprecess kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// Landau-Lifshitz torque without precession
template <typename dataT>
void llnoprecess_fcn(size_t totalThreads, sycl::nd_item<1> item,
                     dataT*  tx, dataT*  ty, dataT*  tz,
                     dataT* mx_, dataT* my_, dataT* mz_,
                     dataT* hx_, dataT* hy_, dataT* hz_,
                     size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
        sycl::vec<dataT, 3> m = {mx_[gid], my_[gid], mz_[gid]};
        sycl::vec<dataT, 3> H = {hx_[gid], hy_[gid], hz_[gid]};

        sycl::vec<dataT, 3> mxH = sycl::cross(m, H);
        sycl::vec<dataT, 3> torque = -sycl::cross(m, mxH);

        tx[gid] = torque.x();
        ty[gid] = torque.y();
        tz[gid] = torque.z();
    }
}

// the function that launches the kernel
template <typename dataT>
void llnoprecess_t(size_t blocks, size_t threads, sycl::queue q,
                   dataT*  tx, dataT*  ty, dataT*  tz,
                   dataT* mx_, dataT* my_, dataT* mz_,
                   dataT* hx_, dataT* hy_, dataT* hz_,
                   size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(llnoprecess_fcn<dataT>, totalThreads, threads,
                             tx, ty, tz,
                             mx_, my_, mz_,
                             hx_, hy_, hz_,
                             N);
}
