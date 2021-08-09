// kernmulc kernel

// device side function. This is essentially the function of the kernel
// Pointwise multiply fftM and fftK (treat both as complex numbers)
template <typename dataT>
void kernmulc_fcn(sycl::nd_item<3> item,
                  dataT* fftM, dataT* fftK,
                  size_t Nx, size_t Ny) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

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
void kernmulc_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                dataT* fftM, dataT* fftK,
                size_t Nx, size_t Ny) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item){
        kernmulc_fcn<dataT>(item,
                            fftM, fftK,
                            Nx, Ny);
    });
}
