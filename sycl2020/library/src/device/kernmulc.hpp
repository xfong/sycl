// kernmulc kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
// Pointwise multiply fftM and fftK (treat both as complex numbers)
template <typename dataT>
inline void kernmulc_fcn(sycl::nd_item<3> item,
                         dataT* fftM, dataT* fftK,
                         size_t   Nx, size_t   Ny) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;

    if ((ix >= Nx) || (iy >= Ny)) {
        return;
    }

    size_t I = iy*Nx + ix;
    size_t e = 2*I;

    dataT reM = fftM[e  ];
    dataT imM = fftM[e+1];
    dataT reK = fftK[e  ];
    dataT imK = fftK[e+1];

    fftM[e  ] = reM * reK - imM * imK;
    fftM[e+1] = reM * imK + imM * reK;
}

// the function that launches the kernel
template <typename dataT>
void kernmulc_t(dim3 blocks, dim3 threads, sycl::queue q,
                dataT* fftM, dataT* fftK,
                size_t Nx, size_t Ny) {
    libMumax3clDeviceFcnCall(kernmulc_fcn<dataT>, blocks, threads,
                             fftM, fftK,
                               Nx,   Ny);
}
