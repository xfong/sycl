#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "stencil.hpp"

template <typename dataT>
class shiftbytesy_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		shiftbytesy_kernel(write_accessor dstPtr,
		             read_accessor srcPtr,
					 size_t Nx,
					 size_t Ny,
					 size_t Nz,
					 size_t shy,
					 dataT clampV)
		    :	dstPtr(dstPtr),
				srcPtr(srcPtr),
				Nx(Nx),
				Ny(Ny),
				Nz(Nz),
				clampV(clampV) {}
		void operator()(sycl::nd_item<1> item) {
			size_t ix = item.get_group(0) * get_num_range(0) + get_local_id(0);
			size_t iy = item.get_group(1) * get_num_range(1) + get_local_id(1);
			size_t iz = item.get_group(2) * get_num_range(2) + get_local_id(2);

			if ((ix < Nx) || (iy < Ny) || (iz < Nz)) {
				size_t iy2 = iy-shy;
				dataT newval;
				if ((iy2 < 0) || (iy2 >= Ny)) {
					newval = clampV;
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
		size_t         shy;
		dataT          clampV;
};

template <typename dataT>
void shiftbytesy_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *src,
                 size_t Nx,
                 size_t Ny,
                 size_t Nz,
                 size_t shy,
                 dataT clampV,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto src_acc = src->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize[2]),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize[2])),
		                 shiftbytesy_kernel<dataT>(dst_acc, src_acc, Nx, Ny, Nz, shy, clampV));
    });
}
