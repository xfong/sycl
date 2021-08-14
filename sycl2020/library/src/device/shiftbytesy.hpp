// shiftbytesy kernel

#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// shift dst by shy cells (positive or negative) along Y-axis.
inline void shiftbytesy_fcn(sycl::nd_item<3> item,
                            uint8_t*    dst,
                            uint8_t*    src,
                            size_t       Nx, size_t Ny, size_t Nz,
                            size_t      shy,
                            uint8_t  clampV) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix < Nx) || (iy < Ny) || (iz < Nz)) {
        size_t iy2 = iy-shy;
        uint8_t newval;
        if ((iy2 < 0) || (iy2 >= Ny)) {
            newval = clampV;
        } else {
            newval = src[idx(ix,iy2,iz)];
        }
        dst[idx(ix,iy,iz)] = newval;
    }

}

// the function that launches the kernel
void shiftbytesy_t(dim3 blocks, dim3 threads, sycl::queue q,
                   uint8_t*    dst,
                   uint8_t*    src,
                   size_t       Nx, size_t Ny, size_t Nz,
                   size_t      shy,
                   uint8_t  clampV) {
    libMumax3clDeviceFcnCall(shiftbytesy_fcn, blocks, threads,
                                dst,
                                src,
                                 Nx, Ny, Nz,
                                shy,
                             clampV);
}
