// normalize2 kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
template <typename dataT>
void normalize2_fcn(size_t totalThreads, sycl::nd_item<1> item,
                    dataT* vx, dataT* vy, dataT* vz,
                    dataT* vol,
                    size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
        dataT v = (vol == NULL) ? (dataT)(1.0) : vol[gid];
        sycl::vec<dataT, 3> V = {v*vx[gid], v*vy[gid], v*vz[gid]};
        V = sycl::normalize(V);
        if (v == (dataT)(0.0)) {
            vx[gid] = (dataT)(0.0);
            vy[gid] = (dataT)(0.0);
            vz[gid] = (dataT)(0.0);
        } else {
            vx[gid] = V.x();
            vy[gid] = V.y();
            vz[gid] = V.z();
        }
    }
}

template <typename dataT>
void normalize2_t(size_t blocks, size_t threads, sycl::queue q,
                  dataT* vx, dataT* vy, dataT* vz,
                  dataT* vol,
                  size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(normalize2_fcn<dataT>, totalThreads, threads,
                             vx, vy, vz,
                             vol,
                             N);
}
