#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
template <typename dataT>
class normalize2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		normalize2_kernel(write_accessor vxPtr,
		             write_accessor vyPtr,
					 write_accessor vzPtr,
					 read_accessor volPtr,
					 size_t N)
		    :	vxPtr(vxPtr),
				vyPtr(vyPtr),
				vzPtr(vzPtr),
				vol(vol),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT v = (vol == NULL) ? (dataT)(1.0) : vol[gid];

				if (v == 0.0) {
					vxPtr[gid] = (dataT)(0.0);
					vyPtr[gid] = (dataT)(0.0);
					vzPtr[gid] = (dataT)(0.0);
				} else {
					dataT VX = v * vxPtr[gid];
					dataT VY = v * vyPtr[gid];
					dataT VZ = v * vzPtr[gid];
					dataT fac = sycl::rsqrt(VX*VX + VY*VY + VZ*VZ);
					vxPtr[gid] = fac*VX;
					vyPtr[gid] = fac*VY;
					vzPtr[gid] = fac*VZ;
				}
			}
		}
	private:
	    write_accessor vxPtr;
	    write_accessor vyPtr;
	    write_accessor vzPtr;
	    read_accessor  vol;
		size_t         N;
};

template <typename dataT>
void normalize2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *vx,
                 sycl::buffer<dataT, 1> *vy,
                 sycl::buffer<dataT, 1> *vz,
                 sycl::buffer<dataT, 1> *vol,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto vx_acc = vx->template get_access<sycl::access::mode::read>(cgh);
        auto vy_acc = vy->template get_access<sycl::access::mode::read>(cgh);
        auto vz_acc = vz->template get_access<sycl::access::mode::read>(cgh);
        auto vol_acc = vol->template get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 normalize2_kernel<dataT>(vx_acc, vy_acc, vz_acc, vol_acc, N));
    });
}
