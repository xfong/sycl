// copypadmul2 kernel

// device side function.This is essentially the function of the kernel
// Copy src (size S, larger) to dst (size D, smaller)
#include "amul.hpp"
#include "constants.hpp"
#include "stencil.hpp"

template <typename dataT>
void copypadmul2_fcn(sycl::nd_item<3> item,
                     dataT* dst,
                     size_t Dx, size_t Dy, size_t Dz,
                     dataT* src,
                     size_t Sx, size_t Sy, size_t Sz,
                     dataT* Ms_, size_t Ms_mul,
                     dataT* vol) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

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
void copypadmul2_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                   dataT* dst,
                   size_t Dx, size_t Dy, size_t Dz,
                   dataT* src,
                   size_t Sx, size_t Sy, size_t Sz,
                   dataT* Ms_, size_t Ms_mul,
                   dataT* vol) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>( blocks[0]*threads[0],  blocks[1]*threads[1],  blocks[2]*threads[2]),
                                     sycl::range<3>(           threads[0],            threads[1],            threads[2])),
        [=](sycl::nd_item<3> item){
        copypadmul2_fcn<dataT>(item,
                        dst,
                        Dx, Dy, Dz,
                        src,
                        Sx, Sy, Sz,
                        Ms_, Ms_mul,
                        vol);
    });
}
