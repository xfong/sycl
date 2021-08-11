// shiftbytesy kernel

#include "include/stencil.hpp"

// device side function. This is essentially the function of the kernel
// shift dst by shy cells (positive or negative) along Y-axis.
template <typename dataT>
void shiftbytesy(sycl::nd_item<3> item,
                 dataT* dst,
                 dataT* src,
                 size_t  Nx, size_t Ny, size_t Nz,
                 size_t shy,
                 dataT clampV) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

    if ((ix < Nx) || (iy < Ny) || (iz < Nz)) {
        size_t iy2 = iy-shy;
        dataT newval;
        if ((iy2 < 0) || (iy2 >= Ny)) {
            newval = clampV;
        } else {
            newval = src[idx(ix2,iy,iz)];
        }
        dst[idx(ix,iy,iz)] = newval;
    }

}

// the function that launches the kernel
template <typename dataT>
void shiftbytesy_t(sizt_t blocks, size_t threads, sycl::queue q,
                   dataT* dst,
                   dataT* src,
                   size_t  Nx, size_t Ny, size_t Nz,
                   size_t shy,
                   dataT clampV) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item) {
            shiftbytesy_fcn<dataT>(dst,
                                   src,
                                    Nx, Ny, Nz,
                                   shy,
                                   clampV);
    });
}
