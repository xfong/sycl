// kernmulrsymm2dz kernel

// device side function. This is essentially the function of the kernel
// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
// Using the same symmetries as kernmulrsymm3d
template <typename dataT>
void kernmulrsymm2dz_fcn(sycl::nd_item<3> item,
                         dataT* fftMz, dataT* fftKzz
                         size_t Nx, size_t Ny) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

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
void kernmulrsymm2dz_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                       dataT* fftMz, dataT* fftKzz,
                       size_t Nx, size_t Ny) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item){
        kernmulrsymm2dz_fcn<dataT>(item,
                                   fftMz, fftKzz,
                                   Nx, Ny);
    });
}
