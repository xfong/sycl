// copypadmul2 kernel

// device side function. This is essentially the function of the kernel
// Copy src (size S, larger) to dst (size D, smaller)
#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

template <typename dataT>
inline void copyunpad_fcn(sycl::nd_item<3> item,
                          dataT* dst,
                          size_t  Dx, size_t Dy, size_t Dz,
                          dataT* src,
                          size_t  Sx, size_t Sy, size_t Sz) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix < Dx) || (iy < Dy) || (iz < Dz)) {
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(ix, iy, iz, Sx, Sy, Sz)];
    }
}

// the function that launches the kernel
template <typename dataT>
void copyunpad_t(dim3 blocks, dim3 threads, sycl::queue q,
                 dataT* dst,
                 size_t  Dx, size_t Dy, size_t Dz,
                 dataT* src,
                 size_t  Sx, size_t Sy, size_t Sz) {
    libMumax3clDeviceFcnCall(copyunpad_fcn<dataT>, blocks, threads,
                             dst,
                              Dx, Dy, Dz,
                             src,
                              Sx, Sy, Sz);
}
