#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// 2D XY (in-plane) micromagnetic kernel multiplication:
// |Mx| = |Kxx Kxy| * |Mx|
// |My|   |Kyx Kyy|   |My|
// Using the same symmetries as kernmulrsymm3d
template <typename dataT>
class kernmulrsymm2dxy_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kernmulrsymm2dxy_kernel(write_accessor fftMxPtr, write_accessor fftMyPtr,
		             read_accessor fftKxxPtr, read_accessor fftKyyPtr, read_accessor fftKxyPtr,
					 size_t Nx, size_t Ny)
		    :	fftMx(fftMxPtr),
			    fftMy(fftMyPtr),
				fftKxx(fftKxxPtr),
				fftKyy(fftKyyPtr),
				fftKxy(fftKxyPtr),
			    Nx(Nx), Ny(Ny) {}
		void operator()(sycl::nd_item<3> item) {
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
	private:
	    write_accessor fftMx;
	    write_accessor fftMy;
	    read_accessor  fftKxx;
	    read_accessor  fftKyy;
	    read_accessor  fftKxy;
		size_t         Nx;
		size_t         Ny;
};

template <typename dataT>
void kernmulrsymm2dxy_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *fftMx, sycl::buffer<dataT, 1> *fftMy,
                 sycl::buffer<dataT, 1> *fftKxx, sycl::buffer<dataT, 1> *fftKyy, sycl::buffer<dataT, 1> *fftKxy,
                 size_t Nx,
                 size_t Ny,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto fftMx_acc = fftMx->template get_access<sycl::access::mode::read_write>(cgh);
        auto fftMy_acc = fftMy->template get_access<sycl::access::mode::read_write>(cgh);
        auto fftKxx_acc = fftKxx->template get_access<sycl::access::mode::read>(cgh);
        auto fftKyy_acc = fftKyy->template get_access<sycl::access::mode::read>(cgh);
        auto fftKxy_acc = fftKxy->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 kernmulrsymm2dxy_kernel<dataT>(fftMx_acc, fftMy_acc, fftKxx_acc, fftKyy_acc, fftKxy_acc, Nx, Ny));
    });
}
