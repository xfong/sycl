#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "stencil.hpp"

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge or clampR at right edge.
template <typename dataT>
class shiftx_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		shiftx_kernel(write_accessor dstPtr,
		             read_accessor srcPtr,
					 size_t Nx,
					 size_t Ny,
					 size_t Nz,
					 size_t shx,
					 dataT clampL,
					 dataT clampR)
		    :	dstPtr(dstPtr),
				srcPtr(srcPtr),
				Nx(Nx),
				Ny(Ny),
				Nz(Nz),
				clampL(clampL),
				clampR(clampR) {}
		void operator()(sycl::nd_item<3> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
			size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

			if ((ix < Nx) || (iy < Ny) || (iz < Nz)) {
				size_t ix2 = ix-shx;
				dataT newval;
				if (ix2 < 0) {
					newval = clampL;
				} else if (ix2 >= Nx) {
					newval = clampR;
				} else {
					newval = srcPtr[idx(ix2,iy,iz)];
				}
				dstPtr[idx(ix,iy,iz)] = newval;
			}
		}
	private:
	    write_accessor dstPtr;
	    read_accessor  srcPtr;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		size_t         shx;
		dataT          clampL;
		dataT          clampR;
};

template <typename dataT>
void shiftx_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *src,
                 size_t Nx,
                 size_t Ny,
                 size_t Nz,
                 size_t shx,
                 dataT clampL,
                 dataT clampR,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto src_acc = src->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize[2]),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize[2])),
		                 shiftx_kernel<dataT>(dst_acc, src_acc, Nx, Ny, Nz, shx, clampL, clampR));
    });
}
