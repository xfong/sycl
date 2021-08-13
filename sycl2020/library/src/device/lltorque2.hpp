// lltorque2 kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// Landau-Lifshitz torque.
#include "include/amul.hpp"

template <typename dataT>
inline void lltorque2_fcn(sycl::nd_item<3> item,
                          dataT*     tx, dataT*        ty, dataT*  tz,
                          dataT*    mx_, dataT*       my_, dataT* mz_,
                          dataT*    hx_, dataT*       hy_, dataT* hz_,
                          dataT* alpha_, dataT  alpha_mul,
                          size_t      N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        sycl::vec<dataT, 3> m = {mx_[gid], my_[gid], mz_[gid]};
        sycl::vec<dataT, 3> H = {hx_[gid], hy_[gid], hz_[gid]};
        dataT alpha = amul<dataT>(alpha_, alpha_mul, gid);

        sycl::vec<dataT, 3> mxH = sycl::cross(m, H);
        dataT gilb = (dataT)(-1.0) / ((dataT)(1.0) + alpha * alpha);
        sycl::vec<dataT, 3> torque = gilb * (mxH + alpha * sycl::cross(m, mxH));

        tx[gid] = torque.x();
        ty[gid] = torque.y();
        tz[gid] = torque.z();
    }
}

// the function that launches the kernel
template <typename dataT>
void lltorque2_t(dim3 blocks, dim3 threads, sycl::queue q,
                   dataT*     tx, dataT*        ty, dataT*  tz,
                   dataT*    mx_, dataT*       my_, dataT* mz_,
                   dataT*    hx_, dataT*       hy_, dataT* hz_,
                   dataT* alpha_, dataT  alpha_mul,
                   size_t      N) {
    libMumax3clDeviceFcnCall(lltorque2_fcn<dataT>, blocks, threads,
                                 tx,        ty, tz,
                                mx_,       my_, mz_,
                                hx_,       hy_, hz_,
                             alpha_, alpha_mul,
                                  N);
}
