#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// 2D Z (out-of-plane only) micromagnetic kernel multiplication:
// Mz = Kzz * Mz
// Using the same symmetries as kernmulrsymm3d
template <typename dataT>
class kernmulrsymm2dz_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kernmulrsymm2dz_kernel(write_accessor fftMzPtr,
		             read_accessor fftKzzPtr,
					 size_t Nx, size_t Ny)
		    :	fftMz(fftMzPtr),
				fftKzz(fftKzzPtr),
			    Nx(Nx), Ny(Ny) {}
		void operator()(sycl::nd_item<3> item) {
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
	private:
	    write_accessor fftMz;
	    read_accessor  fftKzz;
		size_t         Nx;
		size_t         Ny;
};

template <typename dataT>
void kernmulrsymm2dz_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *fftMz,
                 sycl::buffer<dataT, 1> *fftKzz,
                 size_t Nx,
                 size_t Ny,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto fftMz_acc = fftMz->template get_access<sycl::access::mode::read_write>(cgh);
        auto fftKzz_acc = fftKzz->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 kernmulrsymm2dz_kernel<dataT>(fftMz_acc, fftKzz_acc, Nx, Ny));
    });
}
