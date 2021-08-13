// normalize kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function.This is essentially the function of the kernel
// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
template <typename dataT>
inline void normalize_fcn(sycl::nd_item<3> item,
                          dataT*  vx, dataT* vy, dataT* vz,
                          dataT* vol,
                          size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
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
void normalize_t(dim3 blocks, dim3 threads, sycl::queue q,
                  dataT*  vx, dataT* vy, dataT* vz,
                  dataT* vol,
                  size_t   N) {
    libMumax3clDeviceFcnCall(normalize_fcn<dataT>, blocks, threads,
                              vx, vy, vz,
                             vol,
                               N);
}
