#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "stencil.hpp"

// Copy src (size S, larger) to dst (size D, smaller)
template <typename dataT>
class copyunpad_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		copyunpad_kernel(write_accessor dstPtr,
					 size_t Dx,
					 size_t Dy,
					 size_t Dz,
		             read_accessor srcPtr,
					 size_t Sx,
					 size_t Sy,
					 size_t Sz)
		    :	dstPtr(dstPtr),
				Dx(Dx),
				Dy(Dy),
				Dz(Dz),
				srcPtr(srcPtr),
				Sx(Sx),
				Sy(Sy),
				Sz(Sz) {}
		void operator()(sycl::nd_item<3> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
			size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

			if ((ix < Dx) || (iy < Dy) || (iz < Dz)) {
				dstPtr[index(ix, iy, iz,Dx, Dy, Dz)] = src[index(ix, iy, iz, Sx, Sy, Sz)];
			}
		}
	private:
	    write_accessor dstPtr;
		size_t         Dx;
		size_t         Dy;
		size_t         Dz;
	    read_accessor  srcPtr;
		size_t         Sx;
		size_t         Sy;
		size_t         Sz;
};

template <typename dataT>
void copyunpad_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 size_t Dx,
                 size_t Dy,
                 size_t Dz,
                 sycl::buffer<dataT, 1> *src,
                 size_t Sx,
                 size_t Sy,
                 size_t Sz,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto src_acc = src->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize[2]),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize[2])),
		                 copyunpad_kernel<dataT>(dst_acc, Dx, Dy, Dz, src_acc, Sx, Sy, Sz));
    });
}
