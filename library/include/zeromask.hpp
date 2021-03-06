#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// set dst to zero in cells where mask != 0
template <typename dataT>
class zeromask_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		zeromask_kernel(write_accessor dstPtr,
		             read_accessor maskLUTPtr,
					 read_accessor regionsPtr,
					 size_t N)
		    :	dstPtr(dstPtr),
				maskLUTPtr(maskLUTPtr),
				regionsPtr(regionsPtr),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				if (maskLUTPtr[regionsPtr[gid]] != 0) {
					dstPtr[gid] = (dataT)(0.0);
				}
			}
		}
	private:
	    write_accessor dstPtr;
	    read_accessor  maskLUTPtr;
	    read_accessor  regionsPtr;
		size_t         N;
};

template <typename dataT>
void zeromask_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *maskLUT,
                 sycl::buffer<uint8_t, 1> *regions,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto maskLUT_acc = maskLUT->template get_access<sycl::access::mode::read>(cgh);
        auto regions_acc = regions->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 zeromask_kernel<dataT>(dst_acc, maskLUT_acc, regions_acc, N));
    });
}
