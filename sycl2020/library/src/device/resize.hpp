// resize kernel

#include "include/stencil.hpp"


// Select and resize one layer for interactive output
template <typename dataT>
void resize_fcn(sycl::nd_item<3> item,
                dataT* dst,
                size_t Dx, size_t Dy, size_t Dz,
                dataT* src,
                size_t Sx, size_t Sy, size_t Sz,
                int layer,
                int scalex, int scaley) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

    if ((ix < Dx) && (iy < Dy)) {

        dataT sum = (dataT)(0.0);
        dataT n = (dataT)(0.0);

        for (size_t J = 0; J<scaley; J++) {
            size_t j2 = iy*scaley + J;

            for (size_t K = 0; K<scalex; K++) {
                size_t k2 = ix*scalex + K;

                if ((j2 < Sy) && (k2 < Sx)) {
                    sum += src[(layer*Sy + j2)*Sx + k2];
                    n += (dataT)(1.0);
                }
            }
        }
        dst[iy*Dx + ix] = sum / n;
    }
}

template <typename dataT>
void resize_t(size_t blocks[3], size_t threads[3], sycl::queue q,
              dataT*    dst,
              size_t     Dx, size_t Dy, size_t Dz,
              dataT*    src,
              size_t     Sx, size_t Sy, size_t Sz,
              int     layer,
              int    scalex, int scaley) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item) {
            resize_fcn<dataT>(    dst,
                                   Dx,     Dy, Dz,
                                  src,
                                   Sx,     Sy, Sz,
                                layer,
                               scalex, scaley);
    });
}
