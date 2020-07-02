#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class vecnorm_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		vecnorm_kernel(write_accessor dstPtr,
		             read_accessor a0Ptr,
					 read_accessor a1Ptr,
					 read_accessor a2Ptr,
					 size_t N)
		    :	dstPtr(dstPtr),
				axPtr(a0Ptr),
				ayPtr(a1Ptr),
				azPtr(a2Ptr),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				sycl::vec<dataT, 3> A = {axPtr[gid], ayPtr[gid], azPtr[gid]};
				dstPtr[gid] = sycl::sqrt(sycl::dot(A, A));
			}
		}
	private:
	    write_accessor dstPtr;
	    read_accessor  axPtr;
	    read_accessor  ayPtr;
	    read_accessor  azPtr;
		size_t         N;
};

template <typename dataT>
void vecnorm_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *a0,
                 sycl::buffer<dataT, 1> *a1,
                 sycl::buffer<dataT, 1> *a2,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto a0_acc = a0->template get_access<sycl::access::mode::read>(cgh);
        auto a1_acc = a1->template get_access<sycl::access::mode::read>(cgh);
        auto a2_acc = a2->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 vecnorm_kernel<dataT>(dst_acc, a0_acc, a1_acc, a2_acc, N));
    });
}
