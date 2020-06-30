#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// add region-based vector to dst:
// dst[i] += LUT[region[i]]
template <typename dataT>
class regiondecode_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using uint8Acc =
		    sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		regiondecode_kernel(write_accessor dstXPtr, write_accessor dstYPtr, write_accessor dstZPtr,
		             read_accessor srcXPtr, read_accessor srcYPtr, read_accessor srcZPtr,
		             uint8Acc regions,
					 size_t N)
		    :	dstXPtr(dstXPtr),
				dstYPtr(dstYPtr),
				dstZPtr(dstZPtr),
				LUTx(srcXPtr),
				LUTy(srcYPtr),
				LUTz(srcZPtr),
				regions(regions),
				N(N) {}
		void operator()(sycl::nd_item<3> item) {
			size_t gid = (item.get_group(1) * item.get_group_range(0) + item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);

			if (gid < N) {
				uint8_t r = regions[gid];
				dstXPtr[gid] += LUTx[r];
				dstYPtr[gid] += LUTy[r];
				dstZPtr[gid] += LUTz[r];
			}
		}
	private:
	    write_accessor dstXPtr;
	    write_accessor dstYPtr;
	    write_accessor dstZPtr;
	    read_accessor  LUTx;
	    read_accessor  LUTy;
	    read_accessor  LUTz;
		uint8Acc       regions;
		size_t         N;
};

template <typename dataT>
void regiondecode_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dstx,
                 sycl::buffer<dataT, 1> *dsty,
                 sycl::buffer<dataT, 1> *dstz,
                 sycl::buffer<dataT, 1> *LUTx,
                 sycl::buffer<dataT, 1> *LUTy,
                 sycl::buffer<dataT, 1> *LUTz,
                 sycl::buffer<uint8_t, 1> *regions,
                 size_t N,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dstx_acc = dstx->template get_access<sycl::access::mode::read_write>(cgh);
        auto dsty_acc = dsty->template get_access<sycl::access::mode::read_write>(cgh);
        auto dstz_acc = dstz->template get_access<sycl::access::mode::read_write>(cgh);
        auto LUTx_acc = LUTx->template get_access<sycl::access::mode::read>(cgh);
        auto LUTy_acc = LUTy->template get_access<sycl::access::mode::read>(cgh);
        auto LUTz_acc = LUTz->template get_access<sycl::access::mode::read>(cgh);
        auto regions_acc = regions->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 regiondecode_kernel<dataT>(dstx_acc, dsty_acc, dstz_acc, LUTx_acc, LUTy_acc, LUTz_acc, regions_acc, N));
    });
}
