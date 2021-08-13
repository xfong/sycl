// shiftz kernel

#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// shift dst by shz cells (positive or negative) along Z-axis.
// new edge value is clampL at left edge or clampR at right edge.
template <typename dataT>
inline void shiftz_fcn(sycl::nd_item<3> item,
                       dataT*    dst,
                       dataT*    src,
                       size_t     Nx, size_t    Ny, size_t Nz,
                       size_t    shz,
                       dataT  clampL, dataT clampR) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix < Nx) || (iy < Ny) || (iz < Nz)) {
        size_t iz2 = iz-shz;
        dataT newval;
        if (iz2 < 0) {
            newval = clampL;
        } else if (iz2 >= Nz) {
            newval = clampR;
        } else {
            newval = src[idx(ix,iy,iz2)];
        }
        dst[idx(ix,iy,iz)] = newval;
    }
}

// the function that launches the kernel
template <typename dataT>
void shiftz_t(dim3 blocks, dim3 threads, sycl::queue q,
              dataT*    dst,
              dataT*    src,
              size_t     Nx, size_t    Ny, size_t Nz,
              size_t    shz,
              dataT  clampL, dataT clampR) {
    libMumax3clDeviceFcnCall(shiftz_fcn<dataT>, blocks, threads,
                                dst,
                                src,
                                 Nx,     Ny, Nz,
                                shz,
                             clampL, clampR);
}
