// regionaddv kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
// add region-based vector to dst:
// dst[i] += LUT[region[i]]
template <typename dataT>
inline void regionaddv_fcn(sycl::nd_item<3> item,
                           dataT*      dstX, dataT* dstY, dataT* dstZ,
                           dataT*      LUTx, dataT* LUTy, dataT* LUTz,
                           uint8_t* regions,
                           size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        uint8_t r  = regions[gid];
        dstX[gid] += LUTx[r];
        dstY[gid] += LUTy[r];
        dstZ[gid] += LUTz[r];
    }
}

// the function that launches the kernel
template <typename dataT>
void regionaddv_t(dim3 blocks, dim3 threads, sycl::queue q,
                  dataT*      dstx, dataT* dsty, dataT* dstz,
                  dataT*      LUTx, dataT* LUTy, dataT* LUTz,
                  uint8_t* regions,
                  size_t         N) {
    libMumax3clDeviceFcnCall(regionaddv_fcn<dataT>, blocks, threads,
                                dstx, dsty, dstz,
                                LUTx, LUTy, LUTz,
                             regions,
                                   N);
}
