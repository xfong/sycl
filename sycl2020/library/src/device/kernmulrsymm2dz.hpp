// kernmulrsymm2dz kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
// Using the same symmetries as kernmulrsymm3d
template <typename dataT>
inline void kernmulrsymm2dz_fcn(sycl::nd_item<3> item,
                                dataT* fftMz, dataT* fftKzz
                                size_t    Nx, size_t     Ny) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;

    if ((ix >= Nx) || (iy >= Ny)) {
        return;
     }

     size_t I = iy*Nx + ix;
     size_t e = 2 * I;

     dataT reMz = fftMz[e  ];
     dataT imMz = fftMz[e+1];

     if (iy > Ny/2) {
         iy = Ny-iy;
     }
     I = iy*Nx + ix;

     dataT Kzz = fftKzz[I];

     fftMz[e  ] = reMz * Kzz;
     fftMz[e+1] = imMz * Kzz;
}

// the function that launches the kernel
template <typename dataT>
void kernmulrsymm2dz_t(dim3 blocks, dim3 threads, sycl::queue q,
                       dataT* fftMz, dataT* fftKzz,
                       size_t    Nx, size_t     Ny) {
    libMumax3clDeviceFcnCall(kernmulrsymm2dz_fcn<dataT>, blocks, threads,
                             fftMz, fftKzz,
                                Nx,     Ny);
}
