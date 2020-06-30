#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class pointwise_div_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		pointwise_div_kernel(write_accessor dstPtr,
		             read_accessor a0Ptr,
		             read_accessor b0Ptr,
					 size_t N)
		    :	dstPtr(dstPtr),
				axPtr(a0Ptr),
				bxPtr(b0Ptr),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT c0 = bxPtr[gid];
				if (c0 == (dataT)(0.0)) {
					dstPtr[gid] = (dataT)(0.0);
				} else {
					dataT c1 = axPtr[gid];
					if (c1 == (dataT)(0.0)) {
						dstPtr[gid] = (dataT)(0.0);
					} else {
						dstPtr[gid] = c1 / c0;
					}
				}
			}
		}
	private:
	    write_accessor dstPtr;
	    read_accessor  axPtr;
	    read_accessor  bxPtr;
		size_t         N;
};

template <typename dataT>
void pointwise_div_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *a0,
                 sycl::buffer<dataT, 1> *b0,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto a0_acc = a0->template get_access<sycl::access::mode::read>(cgh);
        auto b0_acc = b0->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 pointwise_div_kernel<dataT>(dst_acc, a0_acc, b0_acc, N));
    });
}
