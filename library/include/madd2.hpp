#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class madd2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;
		madd2_kernel(write_accessor dstPtr,
		             read_accessor src1Ptr,
					 dataT fac1,
					 read_accessor src2Ptr,
					 dataT fac2,
					 size_t N)
		    :	dst(dstPtr),
				src1(src1Ptr),
				fac1(fac1),
				src2(src2Ptr),
				fac2(fac2),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dst[gid] = fac1 * src1[gid] + fac2 * src2[gid];
			}
		}
	private:
	    write_accessor dst;
	    read_accessor  src1;
		dataT          fac1;
	    read_accessor  src2;
		dataT          fac2;
		size_t         N;
};

template <typename dataT>
void madd2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *src1,
                 dataT fac1,
                 sycl::buffer<dataT, 1> *src2,
                 dataT fac2,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::write>(cgh);
        auto src1_acc = src1->template get_access<sycl::access::mode::read>(cgh);
        auto src2_acc = src2->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 madd2_kernel<dataT>(dst_acc, src1_acc, fac1, src2_acc, fac2, N));
    });
}
