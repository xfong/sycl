// kernmulrsymm3d kernel

#include "include/utils.h"
#include "include/device_function.hpp"
// device side function. This is essentially the function of the kernel
// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y and Z-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K11, K22, K02:
// xxxxx
// aaaaa
// bbbbb
// ....
// bbbbb
// aaaaa
//
// K12:
// xxxxx
// aaaaa
// bbbbb
// ...
// -bbbb
// -aaaa

template <typename dataT>
inline void kernmulrsymm3d_fcn(sycl::nd_item<3> item,
                               dataT*  fftMx, dataT*  fftMy, dataT*  fftMz,
                               dataT* fftKxx, dataT* fftKyy, dataT* fftKzz,
                               dataT* fftKyz, dataT* fftKxz, dataT* fftKxy,
                               size_t     Nx, size_t     Ny, size_t     Nz) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    // fetch (complex) FFT'ed magnetization
    size_t I = (iz*Ny + iy)*Nx + ix;
    size_t e = 2 * I;
    dataT reMx = fftMx[e  ];
    dataT imMx = fftMx[e+1];
    dataT reMy = fftMy[e  ];
    dataT imMy = fftMy[e+1];
    dataT reMz = fftMz[e  ];
    dataT imMz = fftMz[e+1];

    // fetch kernel

    // minus signs are added to some elements if
    // reconstructed from symmetry.
    dataT signYZ = (dataT)(1.0);
    dataT signXZ = (dataT)(1.0);
    dataT signXY = (dataT)(1.0);

    // use symmetry to fetch from redundant parts:
    // mirror index into first quadrant and set signs.
    if (iy > Ny/2) {
        iy = Ny-iy;
        signYZ = -signYZ;
        signXY = -signXY;
    }
    if (iz > Nz/2) {
        iz = Nz-iz;
        signYZ = -signYZ;
        signXZ = -signXZ;
    }

    // fetch kernel element from non-redundant part
    // and apply minus signs for mirrored parts.
    I = (iz*(Ny/2+1) + iy)*Nx + ix; // Ny/2+1: only half is stored
    dataT Kxx = fftKxx[I];
    dataT Kyy = fftKyy[I];
    dataT Kzz = fftKzz[I];
    dataT Kyz = fftKyz[I] * signYZ;
    dataT Kxz = fftKxz[I] * signXZ;
    dataT Kxy = fftKxy[I] * signXY;

    // m * K matrix multiplication, overwrite m with result.
    fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
    fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
    fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
}

// the function that launches the kernel
template <typename dataT>
void kernmulrsymm3d_t(dim3 blocks, dim3 threads, sycl::queue q,
                dataT*  fftMx, dataT*  fftMy, dataT*  fftMz,
                dataT* fftKxx, dataT* fftKyy, dataT* fftKzz,
                dataT* fftKyz, dataT* fftKxz, dataT* fftKxy,
                size_t     Nx, size_t     Ny, size_t     Nz) {
    libMumax3clDeviceFcnCall(kernmulrsymm3d_fcn<dataT>, blocks, threads,
                                   fftMx,  fftMy,  fftMz,
                                  fftKxx, fftKyy, fftKzz,
                                  fftKyz, fftKxz, fftKxy,
                                      Nx,     Ny,     Nz);
}
