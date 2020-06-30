#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

template <typename dataT>
class crossproduct_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		crossproduct_kernel(write_accessor dstXPtr,
					 write_accessor dstYPtr,
					 write_accessor dstZPtr,
		             read_accessor a0Ptr,
					 read_accessor a1Ptr,
					 read_accessor a2Ptr,
		             read_accessor b0Ptr,
					 read_accessor b1Ptr,
					 read_accessor b2Ptr,
					 size_t N)
		    :	dstXPtr(dstXPtr),
				dstYPtr(dstYPtr),
				dstZPtr(dstZPtr),
				prefactor(prefactor),
				axPtr(a0Ptr),
				ayPtr(a1Ptr),
				azPtr(a2Ptr),
				bxPtr(b0Ptr),
				byPtr(b1Ptr),
				bzPtr(b2Ptr),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT ax = axPtr[gid]; dataT ay = ayPtr[gid]; dataT az = azPtr[gid];
				dataT bx = bxPtr[gid]; dataT by = byPtr[gid]; dataT bz = bzPtr[gid];
				dataT c0 = ay * bz; dataT d0 = by * az;
				dataT c1 = ax * bz; dataT d1 = bx * az;
				dataT c2 = ax * by; dataT d2 = bx * ay;
				ax = c0 - d0;
				ay = d1 - c1;
				az = c2 - d2;
				dstXPtr[gid] = ax;
				dstYPtr[gid] = ay;
				dstZPtr[gid] = az;
			}
		}
	private:
	    write_accessor dstXPtr;
	    write_accessor dstYPtr;
	    write_accessor dstZPtr;
		dataT          prefactor;
	    read_accessor  axPtr;
	    read_accessor  ayPtr;
	    read_accessor  azPtr;
	    read_accessor  bxPtr;
	    read_accessor  byPtr;
	    read_accessor  bzPtr;
		size_t         N;
};

template <typename dataT>
void crossproduct_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dstX,
                 sycl::buffer<dataT, 1> *dstY,
                 sycl::buffer<dataT, 1> *dstZ,
                 sycl::buffer<dataT, 1> *a0,
                 sycl::buffer<dataT, 1> *a1,
                 sycl::buffer<dataT, 1> *a2,
                 sycl::buffer<dataT, 1> *b0,
                 sycl::buffer<dataT, 1> *b1,
                 sycl::buffer<dataT, 1> *b2,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dstX_acc = dstX->template get_access<sycl::access::mode::read_write>(cgh);
        auto dstY_acc = dstY->template get_access<sycl::access::mode::read_write>(cgh);
        auto dstZ_acc = dstZ->template get_access<sycl::access::mode::read_write>(cgh);
        auto a0_acc = a0->template get_access<sycl::access::mode::read>(cgh);
        auto a1_acc = a1->template get_access<sycl::access::mode::read>(cgh);
        auto a2_acc = a2->template get_access<sycl::access::mode::read>(cgh);
        auto b0_acc = b0->template get_access<sycl::access::mode::read>(cgh);
        auto b1_acc = b1->template get_access<sycl::access::mode::read>(cgh);
        auto b2_acc = b2->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 crossproduct_kernel<dataT>(dstX_acc, dstY_acc, dstZ_acc, a0_acc, a1_acc, a2_acc, b0_acc, b1_acc, b2_acc, N));
    });
}
