#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "stencil.hpp"

// Select and resize one layer for interactive output
template <typename dataT>
class resize_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		resize_kernel(write_accessor dstPtr,
					 size_t Dx, size_t Dy, size_t Dz,
		             read_accessor srcPtr,
					 size_t Sx, size_t Sy, size_t Sz,
					 int layer,
					 int scalex, int scaley)
		    :	dstPtr(dstPtr),
			    Dx(Dx), Dy(Dy), Dz(Dz),
				srcPtr(srcPtr),
			    Sx(Sx), Sy(Sy), Sz(Sz),
				layer(layer),
				scalex(scalex),
				scaley(scaley) {}
		void operator()(sycl::nd_item<3> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

			if ((ix < Dx) && (iy < Dy)) {

				dataT sum = (dataT)(0.0);
				dataT n = (dataT)(0.0);

				for (size_t J = 0; J<scaley; J++) {
					size_t j2 = iy*scaley + J;

					for (size_t K = 0; K<scalex; K++) {
						size_t k2 = ix*scalex + K;

						if ((j2 < Sy) && (k2 < Sx)) {
							sum += src[(layer*Sy + j2)*Sx + k2];
							n += (dataT)(1.0);
						}
					}
				}
				dstPtr[iy*Dx + ix] = sum / n;
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
		size_t         layer;
		size_t         scalex;
		size_t         scaley;
};

template <typename dataT>
void resize_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 size_t Dx,
                 size_t Dy,
                 size_t Dz,
                 sycl::buffer<dataT, 1> *src,
                 size_t Sx,
                 size_t Sy,
                 size_t Sz,
                 int    layer,
                 int    scalex,
                 int    scaley,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto src_acc = src->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 resize_kernel<dataT>(dst_acc, Dx, Dy, Dz, src_acc, Sx, Sy, Sz, layer, scalex, scaley));
    });
}
