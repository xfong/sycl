#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// Steepest descent energy minimizer
template <typename dataT>
class minimize_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		minimize_kernel(write_accessor mxPtr, write_accessor myPtr, write_accessor mzPtr,
		             read_accessor m0xPtr, read_accessor m0yPtr, read_accessor m0zPtr,
		             read_accessor txPtr, read_accessor tyPtr, read_accessor tzPtr,
					 dataT dt,
					 size_t N)
		    :	mx_(mxPtr),
				my_(myPtr),
				mz_(mzPtr),
				m0x_(m0xPtr),
				m0y_(m0yPtr),
				m0z_(m0zPtr),
				tx_(txPtr),
				ty_(tyPtr),
				tz_(tzPtr),
				dt(dt),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				sycl::vec<dataT, 3> m0 = {m0x_[gid], m0y_[gid], m0z_[gid]};
				sycl::vec<dataT, 3> t = {tx_[gid], ty_[gid], tz_[gid]};

				sycl::vec<dataT, 3> t2 = dt*dt*sycl::dot(t, t);
				sycl::vec<dataT, 3> result = ((dataT)(4.0) - t2) * m0 + (dataT)(4.0) * dt * t;
				sycl::vec<dataT, 3> divisor = (dataT)(4.0) + t2;
				
				mx_[gid] = result.x() / divisor;
				my_[gid] = result.y() / divisor;
				mz_[gid] = result.z() / divisor;

			}
		}
	private:
	    write_accessor mx_;
	    write_accessor my_;
	    write_accessor mz_;
	    read_accessor  m0x_;
	    read_accessor  m0y_;
	    read_accessor  m0z_;
	    read_accessor  tx_;
	    read_accessor  ty_;
	    read_accessor  tz_;
		dataT          dt;
		size_t         N;
};

template <typename dataT>
void minimize_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *mx,
                 sycl::buffer<dataT, 1> *my,
                 sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *m0x,
                 sycl::buffer<dataT, 1> *m0y,
                 sycl::buffer<dataT, 1> *m0z,
                 sycl::buffer<dataT, 1> *tx,
                 sycl::buffer<dataT, 1> *ty,
                 sycl::buffer<dataT, 1> *tz,
				 dataT dt,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto mx_acc = mx->template get_access<sycl::access::mode::read_write>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read_write>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read_write>(cgh);
        auto m0x_acc = m0x->template get_access<sycl::access::mode::read>(cgh);
        auto m0y_acc = m0y->template get_access<sycl::access::mode::read>(cgh);
        auto m0z_acc = m0z->template get_access<sycl::access::mode::read>(cgh);
        auto tx_acc = tx->template get_access<sycl::access::mode::read>(cgh);
        auto ty_acc = ty->template get_access<sycl::access::mode::read>(cgh);
        auto tz_acc = tz->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 minimize_kernel<dataT>(mx_acc, my_acc, mz_acc, m0x_acc, m0y_acc, m0z_acc, tx_acc, ty_acc, tz_acc, dt, N));
    });
}
