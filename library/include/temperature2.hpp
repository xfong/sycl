#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"

template <typename dataT>
class settemperature2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		settemperature2_kernel(write_accessor BPtr,
		             read_accessor noisePtr, dataT kB2_VgammaDt,
					 read_accessor Ms_, dataT Ms_mul,
					 read_accessor temp_, dataT temp_mul,
					 read_accessor alpha_, dataT alpha_mul,
					 size_t N)
		    :	BPtr(BPtr),
				noisePtr(noisePtr),
				Ms_(Ms_), Ms_mul(Ms_mul),
				temp_(temp_), temp_mul(temp_mul),
				alpha_(alpha_), alpha_mul(alpha_mul),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT invMs = inv_Msat(Ms_, Ms_mul, gid);
				dataT temp = amul(temp_, temp_mul, gid);
				dataT alpha = amul(alpha_, alpha_mul, gid);
				BPtr[gid] = noisePtr[gid] * sycl::srqt(kB2_VgammaDt * alpha * temp * invMs);
			}
		}
	private:
	    write_accessor BPtr;
	    read_accessor  noisePtr;
	    read_accessor  Ms_;
		dataT          Ms_mul;
	    read_accessor  temp_;
		dataT          temp_mul;
	    read_accessor  alpha_;
		dataT          alpha_mul;
		size_t         N;
};

template <typename dataT>
void settemperature2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *B,
                 sycl::buffer<dataT, 1> *noise,
                 dataT kB2_VgammaDt,
                 sycl::buffer<dataT, 1> *Ms, dataT Ms_mul,
                 sycl::buffer<dataT, 1> *temp, dataT temp_mul,
                 sycl::buffer<dataT, 1> *alpha, dataT alpha_mul,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto B_acc = B->template get_access<sycl::access::mode::read_write>(cgh);
        auto noise_acc = noise->template get_access<sycl::access::mode::read>(cgh);
        auto Ms_acc = Ms->template get_access<sycl::access::mode::read>(cgh);
        auto temp_acc = temp->template get_access<sycl::access::mode::read>(cgh);
        auto alpha_acc = alpha->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 settemperature2_kernel<dataT>(B_acc, noise_acc, kB2_VgammaDt, Ms_acc, Ms_mul, temp_acc, temp_mul, alpha_acc, alpha_mul, N));
    });
}
