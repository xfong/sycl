// shiftz kernel

#include "include/stencil.hpp"

// device side function. This is essentially the function of the kernel
// shift dst by shz cells (positive or negative) along Z-axis.
// new edge value is clampL at left edge or clampR at right edge.
template <typename dataT>
void shiftz_fcn(sycl::nd_item<3> item,
                dataT* dst,
                dataT* src,
                size_t Nx, size_t Ny, size_t Nz,
                size_t shz,
                dataT clampL, dataT clampR) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

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
void shiftz_t(size_t blocks[3], size_t threads[3], sycl::queue q,
              dataT* dst,
              dataT* src,
              size_t Nx, size_t Ny, size_t Nz,
              size_t shz,
              dataT clampL, dataT clampR) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item) {
            shiftz_fcn<dataT>(dst,
                              src,
                               Nx, Ny, Nz,
                              shz,
                              clampL, clampR);
    });
}
