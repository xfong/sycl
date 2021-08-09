// kernmulrsymm2dxy kernel

// device side function. This is essentially the function of the kernel
// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
// Using the same symmetries as kernmulrsymm3d
template <typename dataT>
void kernmulrsymm2dxy_fcn(sycl::nd_item<3> item,
                          dataT*  fftMx, dataT*  fftMy,
                          dataT* fftKxx, dataT* fftKyy, dataT* fftKxy,
                          size_t Nx, size_t Ny) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

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
void kernmulrsymm2dxy_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                       dataT*  fftMx, dataT* fftMy,
                       dataT* fftKxx, dataT* fftKyy, dataT* fftKxy,
                       size_t Nx, size_t Ny) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item){
        kernmulrsymm2dxy_fcn<dataT>(item,
                                     fftMx,  fftMy,
                                    fftKxx, fftKyy, fftKxy,
                                    Nx, Ny);
    });
}
