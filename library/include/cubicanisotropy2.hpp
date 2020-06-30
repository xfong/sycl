#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"

template <typename dataT>
class addcubicanisotropy2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		addcubicanisotropy2_kernel(write_accessor BXPtr,
					 write_accessor BYPtr,
					 write_accessor BZPtr,
		             read_accessor mxPtr,
					 read_accessor myPtr,
					 read_accessor mzPtr,
		             read_accessor Ms_, dataT Ms_mul,
		             read_accessor k1_, dataT k1_mul,
		             read_accessor k2_, dataT k2_mul,
		             read_accessor k3_, dataT k3_mul,
		             read_accessor c1xPtr, dataT c1x_mul,
					 read_accessor c1yPtr, dataT c1y_mul,
					 read_accessor c1zPtr, dataT c1z_mul,
		             read_accessor c2xPtr, dataT c2x_mul,
					 read_accessor c2yPtr, dataT c2y_mul,
					 read_accessor c2zPtr, dataT c2z_mul,
					 size_t N)
		    :	BXPtr(BXPtr),
				BYPtr(BYPtr),
				BZPtr(BZPtr),
				mxPtr(mxPtr),
				myPtr(myPtr),
				mzPtr(mzPtr),
				Ms_(Ms_),
				Ms_mul(Ms_mul),
				k1_(k1_),
				k1_mul(k1_mul),
				k2_(k2_),
				k2_mul(k2_mul),
				k3_(k3_),
				k3_mul(k3_mul),
				c1xPtr(c1xPtr),
				c1x_mul(c1x_mul),
				c1yPtr(c1yPtr),
				c1y_mul(c1y_mul),
				c1zPtr(c1zPtr),
				c1z_mul(c1z_mul),
				c2xPtr(c2xPtr),
				c2x_mul(c2x_mul),
				c2yPtr(c2yPtr),
				c2y_mul(c2y_mul),
				c2zPtr(c2zPtr),
				c2z_mul(c2z_mul),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT invMs = inv_Msat(Ms_, Ms_mul, gid);
				dataT k1 = amul(k1_, k1_mul, gid);
				    k1 *= invMs;
				dataT k2 = amul(k2_, k2_mul, gid);
				    k2 *= invMs;
				dataT k3 = amul(k3_, k3_mul, gid);
				    k3 *= invMs;
				dataT u1x = (c1xPtr == NULL) ? c1x_mul : (c1x_mul * c1xPtr[gid]);
				dataT u1y = (c1yPtr == NULL) ? c1y_mul : (c1y_mul * c1yPtr_[gid]);
				dataT u1z = (c1zPtr == NULL) ? c1z_mul : (c1z_mul * c1zPtr_[gid]);
				dataT u1norm = sycl::rsqrt((u1x*u1x) + (u1y*u1y) + (u1z*u1z));
				u1x *= u1norm;
				u1y *= u1norm;
				u1z *= u1norm;
				
				dataT u2x = (c2xPtr == NULL) ? c2x_mul : (c2x_mul * c2xPtr[gid]);
				dataT u2y = (c2yPtr == NULL) ? c2y_mul : (c2y_mul * c2yPtr_[gid]);
				dataT u2z = (c2zPtr == NULL) ? c2z_mul : (c2z_mul * c2zPtr_[gid]);
				dataT u2norm = sycl::rsqrt((u2x*u2x) + (u2y*u2y) + (u2z*u2z));
				u2x *= u2norm;
				u2y *= u2norm;
				u2z *= u2norm;

				dataT u3x = u1y*u2z - u1z*u2y;
				dataT u3y = u1z*u2x - u1x*u2z;
				dataT u3z = u1x*u2y - u1y*u2z;

				dataT u1m = u1x*mxPtr[gid] + u1y*myPtr[gid] + u1z*mzPtr[gid];
				dataT u2m = u2x*mxPtr[gid] + u2y*myPtr[gid] + u2z*mzPtr[gid];
				dataT u3m = u3x*mxPtr[gid] + u3y*myPtr[gid] + u3z*mzPtr[gid];
				
				dataT u1m2 = pow2(u1m); dataT u2m2 = pow2(u2m); dataT u3m2 = pow2(u3m);
				dataT u1m4 = pow2(u1m2); dataT u2m4 = pow2(u2m2); dataT u3m4 = pow2(u3m2);

				dataT tmp_x = (dataT)(-2.0)*k1*((u2m2 + u3m2) * (    (u1m) * u1x) +
				                                (u1m2 + u3m2) * (    (u2m) * u2x) +
										        (u1m2 + u2m2) * (    (u3m) * u3x))-
							   (dataT)(2.0)*k2*((u2m2 * u3m2) * (    (u1m) * u1x) +
							                    (u1m2 * u3m2) * (    (u2m) * u2x) +
											    (u1m2 * u2m2) * (    (u3m) * u3x))-
							   (dataT)(4.0)*k3*((u2m4 + u3m4) * (pow3(u1m) * u1x) +
							                    (u1m4 + u3m4) * (pow3(u2m) * u2x) +
							                    (u1m4 + u2m4) * (pow3(u3m) * u3x));
				dataT tmp_y = (dataT)(-2.0)*k1*((u2m2 + u3m2) * (    (u1m) * u1y) +
				                                (u1m2 + u3m2) * (    (u2m) * u2y) +
										        (u1m2 + u2m2) * (    (u3m) * u3y))-
							   (dataT)(2.0)*k2*((u2m2 * u3m2) * (    (u1m) * u1y) +
							                    (u1m2 * u3m2) * (    (u2m) * u2y) +
											    (u1m2 * u2m2) * (    (u3m) * u3y))-
							   (dataT)(4.0)*k3*((u2m4 + u3m4) * (pow3(u1m) * u1y) +
							                    (u1m4 + u3m4) * (pow3(u2m) * u2y) +
							                    (u1m4 + u2m4) * (pow3(u3m) * u3y));
				dataT tmp_z = (dataT)(-2.0)*k1*((u2m2 + u3m2) * (    (u1m) * u1z) +
				                                (u1m2 + u3m2) * (    (u2m) * u2z) +
										        (u1m2 + u2m2) * (    (u3m) * u3z))-
							   (dataT)(2.0)*k2*((u2m2 * u3m2) * (    (u1m) * u1z) +
							                    (u1m2 * u3m2) * (    (u2m) * u2z) +
											    (u1m2 * u2m2) * (    (u3m) * u3z))-
							   (dataT)(4.0)*k3*((u2m4 + u3m4) * (pow3(u1m) * u1z) +
							                    (u1m4 + u3m4) * (pow3(u2m) * u2z) +
							                    (u1m4 + u2m4) * (pow3(u3m) * u3z));

				// Store to global buffer
				BXPtr[gid] = tmp_x;
				BYPtr[gid] = tmp_y;
				BZPtr[gid] = tmp_z;
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
	    read_accessor  k3_;
		dataT          k3_mul;
	    read_accessor  c1xPtr;
	    read_accessor  c1yPtr;
	    read_accessor  c1zPtr;
	    read_accessor  c2xPtr;
	    read_accessor  c2yPtr;
	    read_accessor  c2zPtr;
		size_t         N;
};

template <typename dataT>
void addcubicanisotropy2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *BX,
                 sycl::buffer<dataT, 1> *BY,
                 sycl::buffer<dataT, 1> *BZ,
                 sycl::buffer<dataT, 1> *mx,
                 sycl::buffer<dataT, 1> *my,
                 sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *Ms_, dataT Ms_mul,
                 sycl::buffer<dataT, 1> *k1_, dataT k1_mul,
                 sycl::buffer<dataT, 1> *k2_, dataT k2_mul,
				 sycl::buffer<dataT, 1> *k3_, dataT k3_mul,
                 sycl::buffer<dataT, 1> *c1x_, dataT c1x_mul,
                 sycl::buffer<dataT, 1> *c1y_, dataT c1y_mul,
                 sycl::buffer<dataT, 1> *c1z_, dataT c1z_mul,
                 sycl::buffer<dataT, 1> *c2x_, dataT c2x_mul,
                 sycl::buffer<dataT, 1> *c2y_, dataT c2y_mul,
                 sycl::buffer<dataT, 1> *c2z_, dataT c2z_mul,
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
        auto k3_acc = k3_->template get_access<sycl::access::mode::read>(cgh);
        auto c1x_acc = c1x_->template get_access<sycl::access::mode::read>(cgh);
        auto c1y_acc = c1y_->template get_access<sycl::access::mode::read>(cgh);
        auto c1z_acc = c1z_->template get_access<sycl::access::mode::read>(cgh);
        auto c2x_acc = c2x_->template get_access<sycl::access::mode::read>(cgh);
        auto c2y_acc = c2y_->template get_access<sycl::access::mode::read>(cgh);
        auto c2z_acc = c2z_->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 addcubicanisotropy2_kernel<dataT>(BX_acc, BY_acc, BZ_acc,
														   mx_acc, my_acc, mz_acc,
						                                   Ms_acc, Ms_mul,
														   k1_acc, k1_mul,
														   k2_acc, k2_mul,
														   k3_acc, k3_mul,
														   c1x_acc, c1x_mul,
														   c1y_acc, c1y_mul,
														   c1z_acc, c1z_mul,
														   c2x_acc, c2x_mul,
														   c2y_acc, c2y_mul,
														   c2z_acc, c2z_mul,
														   N));
    });
}
