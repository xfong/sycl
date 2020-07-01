#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class kernmulc_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kernmulc_kernel(write_accessor fftMPtr,
		             read_accessor fftKPtr,
					 size_t Nx, size_t Ny)
		    :	fftM(fftMPtr),
				fftK(fftKPtr),
			    Nx(Nx), Ny(Ny) {}
		void operator()(sycl::nd_item<3> item) {
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
	private:
	    write_accessor fftM;
	    read_accessor  fftK;
		size_t         Nx;
		size_t         Ny;
\};

template <typename dataT>
void kernmulc_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *fftM,
                 sycl::buffer<dataT, 1> *fftK,
                 size_t Nx,
                 size_t Ny,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto fftM_acc = fftM->template get_access<sycl::access::mode::read_write>(cgh);
        auto fftK_acc = fftK->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 kernmulc_kernel<dataT>(fftM_acc, fftK_acc, Nx, Ny));
    });
}
