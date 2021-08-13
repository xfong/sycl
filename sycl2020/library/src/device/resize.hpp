// resize kernel

#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// Select and resize one layer for interactive output
template <typename dataT>
inline void resize_fcn(sycl::nd_item<3> item,
                       dataT*    dst,
                       size_t     Dx, size_t  Dy, size_t Dz,
                       dataT*    src,
                       size_t     Sx, size_t  Sy, size_t Sz,
                       int     layer,
                       int    scalex, int scaley) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;

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
void resize_t(dim3 blocks, dim3 threads, sycl::queue q,
              dataT*    dst,
              size_t     Dx, size_t Dy, size_t Dz,
              dataT*    src,
              size_t     Sx, size_t Sy, size_t Sz,
              int     layer,
              int    scalex, int scaley) {
    libMumax3clDeviceFcnCall(resize_fcn<dataT>, blocks, threads,
                                dst,
                                 Dx,     Dy, Dz,
                                src,
                                 Sx,     Sy, Sz,
                              layer,
                             scalex, scaley);
}
