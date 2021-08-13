// exchangedecode kernel

#include "include/amul.hpp"
#include "include/exchange.hpp"
#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// Finds the average exchange strength around each cell, for debugging.
template <typename dataT>
inline void exchangedecode_fcn(sycl::nd_item<3> item,
                               dataT*       dst,
                               dataT*    aLUT2d,
                               uint8_t* regions,
                               dataT         wx, dataT  wy, dataT  wz,
                               size_t        Nx, size_t Ny, size_t Nz,
                               uint8_t      PBC) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    //central cell
    size_t I   = idx(ix, iy, iz);
    uint8_t r0 = regions[I];

    size_t i_; // neighbor index
    dataT avg = (dataT)(0.0);

    // left neighbor
    i_   = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    avg += aLUT2d[symidx(r0, regions[i_])];

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    avg = aLUT2d[symidx(r0, regions[i_])];

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    avg = aLUT2d[symidx(r0, regions[i_])];

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    avg = aLUT2d[symidx(r0, regions[i_])];

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        avg = aLUT2d[symidx(r0, regions[i_])];

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        avg = aLUT2d[symidx(r0, regions[i_])];
    }

    dst[I] = avg;

}

// the function that launches the kernel
template <typename dataT>
void exchangedecode_t(dim3 blocks, dim3 threads, sycl::queue q,
                      dataT*       dst,
                      dataT*    aLUT2d,
                      uint8_t* regions,
                      dataT         wx, dataT  wy, dataT  wz,
                      size_t        Nx, size_t Ny, size_t Nz,
                      uint8_t      PBC) {
    libMumax3clDeviceFcnCall(exchangedecode_fcn<dataT>, blocks, threads,
                                 dst,
                              aLUT2d,
                             regions,
                                  wx, wy, wz,
                                  Nx, Ny, Nz,
                                 PBC);
}
