// copypadmul2 kernel

// device side function. This is essentially the function of the kernel
// Copy src (size S, larger) to dst (size D, smaller)
#include "include/amul.hpp"
#include "include/constants.hpp"
#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

template <typename dataT>
inline void copypadmul2_fcn(sycl::nd_item<3> item,
                            dataT* dst,
                            size_t  Dx, size_t     Dy, size_t Dz,
                            dataT* src,
                            size_t  Sx, size_t     Sy, size_t Sz,
                            dataT* Ms_, size_t Ms_mul,
                            dataT* vol) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix < Sx) || (iy < Sy) || (iz < Sz)) {
        size_t sI = index(ix, iy, iz, Sx, Sy, Sz);  // source index
        dataT tmpFac = amul<dataT>(Ms_, Ms_mul, sI);
        dataT Bsat = MU0 * tmpFac;
        dataT v = amul<dataT>(vol, (dataT)(1.0), sI);
        dst[index(ix, iy, iz,Dx, Dy, Dz)] = Bsat * v * src[sI];
    }
}

// the function that launches the kernel
template <typename dataT>
void copypadmul2_t(dim3 blocks, dim3 threads, sycl::queue q,
                   dataT* dst,
                   size_t  Dx, size_t     Dy, size_t Dz,
                   dataT* src,
                   size_t  Sx, size_t     Sy, size_t Sz,
                   dataT* Ms_, size_t Ms_mul,
                   dataT* vol) {
    libMumax3clDeviceFcnCall(copypadmul2_fcn<dataT>, blocks, threads,
                             dst,
                              Dx, Dy, Dz,
                             src,
                              Sx, Sy, Sz,
                             Ms_, Ms_mul,
                             vol);
}
