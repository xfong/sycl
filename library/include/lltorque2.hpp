#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"

// Landau-Lifshitz torque.
template <typename dataT>
class llnoprecess_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		llnoprecess_kernel(write_accessor tXPtr, write_accessor tYPtr, write_accessor tZPtr,
		             read_accessor mxPtr, read_accessor myPtr, read_accessor mzPtr,
		             read_accessor hxPtr, read_accessor hyPtr, read_accessor hzPtr,
					 read_accessor alphaPtr, dataT alpha_mul,
					 size_t N)
		    :	tx(tXPtr),
				ty(tYPtr),
				tz(tZPtr),
				mx_(mxPtr),
				my_(myPtr),
				mz_(mzPtr),
				hx_(hxPtr),
				hy_(hyPtr),
				hz_(hzPtr),
				alpha_(alphaPtr),
				alpha_mul(alpha_mul),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				sycl::vec<dataT, 3> m = {mx_[gid], my_[gid], mz_[gid]};
				sycl::vec<dataT, 3> H = {hx_[gid], hy_[gid], hz_[gid]};
				dataT alpha = amul(alpha_, alpha_mul, i);

				sycl::vec<dataT, 3> mxH = sycl::cross(m, H);
				dataT gilb = (dataT)(-1.0) / ((dataT)(1.0) + alpha * alpha);
				sycl::vec<dataT, 3> torque = gilb * (mxH + alpha * sycl::cross(m, mxH));

				tx[gid] = torque.x();
				ty[gid] = torque.y();
				tz[gid] = torque.z();
			}
		}
	private:
	    write_accessor tx;
	    write_accessor ty;
	    write_accessor tz;
	    read_accessor  mx_;
	    read_accessor  my_;
	    read_accessor  mz_;
	    read_accessor  hx_;
	    read_accessor  hy_;
	    read_accessor  hz_;
		read_accessor  alpha_;
		dataT          alpha_mul;
		size_t         N;
};

template <typename dataT>
void llnoprecess_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *tx,
                 sycl::buffer<dataT, 1> *ty,
                 sycl::buffer<dataT, 1> *tz,
                 sycl::buffer<dataT, 1> *mx,
                 sycl::buffer<dataT, 1> *my,
                 sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *hx,
                 sycl::buffer<dataT, 1> *hy,
                 sycl::buffer<dataT, 1> *hz,
                 sycl::buffer<dataT, 1> *alpha_,
				 dataT alpha_mul,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto tx_acc = tx->template get_access<sycl::access::mode::read_write>(cgh);
        auto ty_acc = ty->template get_access<sycl::access::mode::read_write>(cgh);
        auto tz_acc = tz->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto hx_acc = hx->template get_access<sycl::access::mode::read>(cgh);
        auto hy_acc = hy->template get_access<sycl::access::mode::read>(cgh);
        auto hz_acc = hz->template get_access<sycl::access::mode::read>(cgh);
        auto alpha_acc = alpha_->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 llnoprecess_kernel<dataT>(tx_acc, ty_acc, tz_acc, mx_acc, my_acc, mz_acc, hx_acc, hy_acc, hz_acc, alpha_acc, alpha_mul, N));
    });
}
