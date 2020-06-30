#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"

// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
template <typename dataT>
class adduniaxialanisotropy2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		adduniaxialanisotropy2_kernel(write_accessor BXPtr,
					 write_accessor BYPtr,
					 write_accessor BZPtr,
		             read_accessor mxPtr,
					 read_accessor myPtr,
					 read_accessor mzPtr,
		             read_accessor Ms_, dataT Ms_mul,
		             read_accessor k1_, dataT k1_mul,
		             read_accessor k2_, dataT k2_mul,
		             read_accessor uxPtr, dataT ux_mul,
					 read_accessor uyPtr, dataT uy_mul,
					 read_accessor uzPtr, dataT uz_mul,
					 size_t N)
		    :	BXPtr(BXPtr),
				BYPtr(BYPtr),
				BZPtr(BZPtr),
				mxPtr(mxPtr),
				myPtr(myPtr),
				mzPtr(mzPtr),
				k1_(k1_),
				k1_mul(k1_mul),
				k2_(k2_),
				k2_mul(k2_mul),
				uxPtr(uxPtr),
				ux_mul(ux_mul),
				uyPtr(uyPtr),
				uy_mul(uy_mul),
				uzPtr(uzPtr),
				uz_mul(uz_mul),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT invMs = inv_Msat(Ms_, Ms_mul, gid);
				dataT k1 = amul(k1_, k1_mul, gid);
				    k1 *= invMs;
				dataT k2 = amul(k2_, k2_mul, gid);
				    k2 *= invMs;
				dataT u1x = (uxPtr == NULL) ? ux_mul : (ux_mul * uxPtr[gid]);
				dataT u1y = (uyPtr == NULL) ? uy_mul : (uy_mul * uyPtr_[gid]);
				dataT u1z = (uzPtr == NULL) ? uz_mul : (uz_mul * uzPtr_[gid]);
				dataT u1norm = sycl::rsqrt((u1x*u1x) + (u1y*u1y) + (u1z*u1z));
				u1x *= u1norm;
				u1y *= u1norm;
				u1z *= u1norm;
				
				dataT u1m = u1x*mxPtr[gid] + u1y*myPtr[gid] + u1z*mzPtr[gid];
				
				dataT u1m3 = pow3(u1m);

				dataT tmp_x = (dataT)(2.0)*k1*(u1m * u1x) +
							  (dataT)(4.0)*k2*(u1m3 * u1x);

				dataT tmp_y = (dataT)(2.0)*k1*(u1m * u1y) +
							  (dataT)(4.0)*k2*(u1m3 * u1y);

				dataT tmp_z = (dataT)(2.0)*k1*(u1m * u1z) +
							  (dataT)(4.0)*k2*(u1m3 * u1z);

				// Store to global buffer
				BXPtr[gid] += tmp_x;
				BYPtr[gid] += tmp_y;
				BZPtr[gid] += tmp_z;
			}
		}
	private:
	    write_accessor BXPtr;
	    write_accessor BYPtr;
	    write_accessor BZPtr;
	    read_accessor  mxPtr;
	    read_accessor  myPtr;
	    read_accessor  mzPtr;
	    read_accessor  Ms_;
		dataT          Ms_mul;
	    read_accessor  k1_;
		dataT          k1_mul;
	    read_accessor  k2_;
		dataT          k2_mul;
	    read_accessor  uxPtr;
		dataT          ux_mul;
	    read_accessor  uyPtr;
		dataT          uy_mul;
	    read_accessor  uzPtr;
		dataT          uz_mul;
		size_t         N;
};

template <typename dataT>
void adduniaxialanisotropy2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *BX,
                 sycl::buffer<dataT, 1> *BY,
                 sycl::buffer<dataT, 1> *BZ,
                 sycl::buffer<dataT, 1> *mx,
                 sycl::buffer<dataT, 1> *my,
                 sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *Ms_, dataT Ms_mul,
                 sycl::buffer<dataT, 1> *k1_, dataT k1_mul,
                 sycl::buffer<dataT, 1> *k2_, dataT k2_mul,
                 sycl::buffer<dataT, 1> *ux_, dataT ux_mul,
                 sycl::buffer<dataT, 1> *uy_, dataT uy_mul,
                 sycl::buffer<dataT, 1> *uz_, dataT uz_mul,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto BX_acc = BX->template get_access<sycl::access::mode::read_write>(cgh);
        auto BY_acc = BY->template get_access<sycl::access::mode::read_write>(cgh);
        auto BZ_acc = BZ->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto Ms_acc = Ms_->template get_access<sycl::access::mode::read>(cgh);
        auto k1_acc = k1_->template get_access<sycl::access::mode::read>(cgh);
        auto k2_acc = k2_->template get_access<sycl::access::mode::read>(cgh);
        auto ux_acc = ux_->template get_access<sycl::access::mode::read>(cgh);
        auto uy_acc = uy_->template get_access<sycl::access::mode::read>(cgh);
        auto uz_acc = uz_->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 adduniaxialanisotropy2_kernel<dataT>(BX_acc, BY_acc, BZ_acc,
														   mx_acc, my_acc, mz_acc,
						                                   Ms_acc, Ms_mul,
														   k1_acc, k1_mul,
														   k2_acc, k2_mul,
														   ux_acc, ux_mul,
														   uy_acc, uy_mul,
														   uz_acc, uz_mul,
														   N));
    });
}
