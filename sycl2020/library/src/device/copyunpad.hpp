// copypadmul2 kernel

#include "device_function.hpp"

// device side function.This is essentially the function of the kernel
// Copy src (size S, larger) to dst (size D, smaller)
#include "stencil.hpp"

template <typename dataT>
void copyunpad_fcn(sycl::nd_item<3> item,
                   dataT* dst,
                   size_t Dx, size_t Dy, size_t Dz,
                   dataT* src,
                   size_t Sx, size_t Sy, size_t Sz) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

    if ((ix < Dx) || (iy < Dy) || (iz < Dz)) {
        dst[index(ix, iy, iz, Dx, Dy, Dz)] = src[index(ix, iy, iz, Sx, Sy, Sz)];
    }
}

// the function that launches the kernel
template <typename dataT>
void copyunpad_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                 dataT* dst,
                 size_t Dx, size_t Dy, size_t Dz,
                 dataT* src,
                 size_t Sx, size_t Sy, size_t Sz) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item){
        copyunpad_fcn<dataT>(item,
                             dst,
                             Dx, Dy, Dz,
                             src,
                             Sx, Sy, Sz);
    });
}
