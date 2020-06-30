#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class square_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		square_kernel(write_accessor dstPtr,
		             read_accessor srcPtr,
					 size_t N)
		    :	dstPtr(dstPtr),
				srcPtr(srcPtr),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT c0 = srcPtr[gid];
				dstPtr[gid] = c0 * c0;
			}
		}
	private:
	    write_accessor dstPtr;
	    read_accessor  srcPtr;
		size_t         N;
};

template <typename dataT>
void square_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *src,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto src_acc = src->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 square_kernel<dataT>(dst_acc, src_acc, N));
    });
}
