// kernmulrsymm2dxy kernel

#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
// Using the same symmetries as kernmulrsymm3d
template <typename dataT>
inline void kernmulrsymm2dxy_fcn(sycl::nd_item<3> item,
                                 dataT*  fftMx, dataT*  fftMy,
                                 dataT* fftKxx, dataT* fftKyy, dataT* fftKxy,
                                 size_t     Nx, size_t     Ny) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;

    if ((ix >= Nx) || (iy >= Ny)) {
        return;
    }

    size_t I = iy*Nx + ix;
    size_t e = 2 * I;

    dataT reMx = fftMx[e  ];
    dataT imMx = fftMx[e+1];
    dataT reMy = fftMy[e  ];
    dataT imMy = fftMy[e+1];

    // symmetry factor
    dataT fxy = (dataT)(1.0);
    if (iy > Ny/2) {
        iy = Ny-iy;
        fxy = -fxy;
    }
    I = iy*Nx + ix;

    dataT Kxx = fftKxx[I];
    dataT Kyy = fftKyy[I];
    dataT Kxy = fxy * fftKxy[I];

    fftMx[e  ] = reMx * Kxx + reMy * Kxy;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy;
}

// the function that launches the kernel
template <typename dataT>
void kernmulrsymm2dxy_t(dim3 blocks, dim3 threads, sycl::queue q,
                        dataT*  fftMx, dataT*  fftMy,
                        dataT* fftKxx, dataT* fftKyy, dataT* fftKxy,
                        size_t     Nx, size_t     Ny) {
    libMumax3clDeviceFcnCall(kernmulrsymm2dxy_fcn<dataT>, blocks, threads,
                              fftMx,  fftMy,
                             fftKxx, fftKyy, fftKxy,
                                 Nx,     Ny);
}
