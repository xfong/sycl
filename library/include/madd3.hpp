#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class madd3_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		madd3_kernel(write_accessor dstPtr,
		             read_accessor src1Ptr,
					 dataT fac1,
					 read_accessor src2Ptr,
					 dataT fac2,
					 read_accessor src3Ptr,
					 dataT fac3,
					 size_t N)
		    :	dstPtr(dstPtr),
				src1Ptr(src1Ptr),
				fac1(fac1),
				src2Ptr(src2Ptr),
				fac2(fac2),
				src3Ptr(src3Ptr),
				fac3(fac3),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dstPtr[gid] = (fac1 * src1Ptr[gid]) + (fac2 * src2Ptr[gid] + fac3 * src3Ptr[gid]);
				// parens for better accuracy heun solver
			}
		}
	private:
	    write_accessor dstPtr;
	    read_accessor  src1Ptr;
		dataT          fac1;
	    read_accessor  src2Ptr;
		dataT          fac2;
	    read_accessor  src3Ptr;
		dataT          fac3;
		size_t         N;
};

template <typename dataT>
void madd3_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *src1,
                 dataT fac1,
                 sycl::buffer<dataT, 1> *src2,
                 dataT fac2,
                 sycl::buffer<dataT, 1> *src3,
                 dataT fac3,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto src1_acc = src1->template get_access<sycl::access::mode::read>(cgh);
        auto src2_acc = src2->template get_access<sycl::access::mode::read>(cgh);
        auto src3_acc = src3->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 madd3_kernel<dataT>(dst_acc, src1_acc, fac1, src2_acc, fac2, src3_acc, fac3, N));
    });
}
