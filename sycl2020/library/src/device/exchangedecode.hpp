// exchangedecode kernel

#include "amul.hpp"
#include "exchange.hpp"
#include "stencil.hpp"

// device side function.This is essentially the function of the kernel
// Finds the average exchange strength around each cell, for debugging.
template <typename dataT>
void exchangedecode_fcn(sycl::nd_item<3> item,
                        dataT* dst,
                        dataT* aLUT2d,
                        uint8_t* regions,
                        dataT wx, dataT wy, dataT wz,
                        size_t Nx, size_t Ny, size_t Nz,
                        uint8_t PBC) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

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
void exchangedecode_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                        dataT* dst,
                        dataT* aLUT2d,
                        uint8_t* regions,
                        dataT wx, dataT wy, dataT wz,
                        size_t Nx, size_t Ny, size_t Nz,
                        uint8_t PBC) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=] (sycl::nd_item<3> item){
        exchangedecode_fcn<dataT>(item,
                        dst,
                        aLUT2d,
                        regions,
                        wx, wy, wz,
                        Nx, Ny, Nz,
                        PBC);
    });
}
