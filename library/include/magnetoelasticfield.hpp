#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"
#include "constants.hpp"

// Add magneto-elastic coupling field to B.
// H = - δUmel / δM, 
// where Umel is magneto-elastic energy denstiy given by the eq. (12.18) of Gurevich&Melkov "Magnetization Oscillations and Waves", CRC Press, 1996
template <typename dataT>
class getmagnetoelasticfield_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		getmagnetoelasticfield_kernel(write_accessor BxPtr, write_accessor ByPtr, write_accessor BzPtr,
		             read_accessor mxPtr, read_accessor myPtr, read_accessor mzPtr,
		             read_accessor exxPtr, dataT exx_mul,
		             read_accessor eyyPtr, dataT eyy_mul,
		             read_accessor ezzPtr, dataT ezz_mul,
		             read_accessor exyPtr, dataT exy_mul,
		             read_accessor exzPtr, dataT exz_mul,
		             read_accessor eyzPtr, dataT eyz_mul,
		             read_accessor B1Ptr, dataT B1_mul,
					 read_accessor B2Ptr, dataT B2_mul,
		             read_accessor MsPtr, dataT Ms_mul,
					 size_t N)
		    :	Bx(BxPtr),
				By(ByPtr),
				Bz(BzPtr),
				mx(mxPtr),
				my(myPtr),
				mz(mzPtr),
				exx_(exxPtr),
				exx_mul(exx_mul),
				eyy_(eyyPtr),
				eyy_mul(eyy_mul),
				ezz_(ezzPtr),
				ezz_mul(ezz_mul),
				exy_(exyPtr),
				exy_mul(exy_mul),
				exz_(exzPtr),
				exz_mul(exz_mul),
				eyz_(eyzPtr),
				eyz_mul(eyz_mul),
				B1_(B1Ptr),
				B1_mul(B1_mul),
				B2_(B2Ptr),
				B2_mul(B2_mul),
				Ms_(MsPtr),
				Ms_mul(Ms_mul),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {

				dataT Exx = amul(exx_, exx_mul, gid);
				dataT Eyy = amul(eyy_, eyy_mul, gid);
				dataT Ezz = amul(ezz_, ezz_mul, gid);
				
				dataT Exy = amul(exy_, exy_mul, gid);
				dataT Eyx = Exy;

				dataT Exz = amul(exz_, exz_mul, gid);
				dataT Ezx = Exz;

				dataT Eyz = amul(eyz_, eyz_mul, gid);
				dataT Ezy = Eyz;

				dataT invMs = inv_Msat(Ms_, Ms_mul, gid);

				dataT B1 = amul(B1_, B1_mul, gid) * invMs;
				dataT B2 = amul(B2_, B2_mul, gid) * invMs;

				sycl::vec<dataT, 3> m = {mx[gid], my[gid], mz[gid]};

				Bx[I] += -((dataT)(2.0)*B1*m.x()*Exx + B2*(m.y()*Exy + m.z()*Exz));
				By[I] += -((dataT)(2.0)*B1*m.y()*Eyy + B2*(m.x()*Eyx + m.z()*Eyz));
				Bz[I] += -((dataT)(2.0)*B1*m.z()*Ezz + B2*(m.x()*Ezx + m.y()*Ezy));
			}
		}
	private:
	    write_accessor Bx;
	    write_accessor By;
	    write_accessor Bz;
	    read_accessor  mx;
	    read_accessor  my;
	    read_accessor  mz;
	    read_accessor  exx_;
		dataT          exx_mul;
	    read_accessor  eyy_;
		dataT          eyy_mul;
	    read_accessor  ezz_;
		dataT          ezz_mul;
	    read_accessor  exy_;
		dataT          exy_mul;
	    read_accessor  exz_;
		dataT          exz_mul;
	    read_accessor  eyz_;
		dataT          eyz_mul;
	    read_accessor  B1_;
		dataT          B1_mul;
	    read_accessor  B2_;
		dataT          B2_mul;
	    read_accessor  Ms_;
		dataT          Ms_mul;
		size_t         N;
};

template <typename dataT>
void getmagnetoelasticfield_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *Bx, sycl::buffer<dataT, 1> *By, sycl::buffer<dataT, 1> *Bz,
                 sycl::buffer<dataT, 1> *mx, sycl::buffer<dataT, 1> *my, sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *exx, dataT exx_mul,
                 sycl::buffer<dataT, 1> *eyy, dataT eyy_mul,
                 sycl::buffer<dataT, 1> *ezz, dataT ezz_mul,
                 sycl::buffer<dataT, 1> *exy, dataT exy_mul,
                 sycl::buffer<dataT, 1> *exz, dataT exz_mul,
                 sycl::buffer<dataT, 1> *eyz, dataT eyz_mul,
                 sycl::buffer<dataT, 1> *B1, dataT B1_mul,
                 sycl::buffer<dataT, 1> *B2, dataT B2_mul,
                 sycl::buffer<dataT, 1> *Ms, dataT Ms_mul,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto Bx_acc = Bx->template get_access<sycl::access::mode::read_write>(cgh);
        auto By_acc = By->template get_access<sycl::access::mode::read_write>(cgh);
        auto Bz_acc = Bz->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto exx_acc = exx->template get_access<sycl::access::mode::read>(cgh);
        auto eyy_acc = eyy->template get_access<sycl::access::mode::read>(cgh);
        auto ezz_acc = ezz->template get_access<sycl::access::mode::read>(cgh);
        auto exy_acc = exy->template get_access<sycl::access::mode::read>(cgh);
        auto exz_acc = exz->template get_access<sycl::access::mode::read>(cgh);
        auto eyz_acc = eyz->template get_access<sycl::access::mode::read>(cgh);
        auto B1_acc = B1->template get_access<sycl::access::mode::read>(cgh);
        auto B2_acc = B2->template get_access<sycl::access::mode::read>(cgh);
        auto Ms_acc = Ms->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 getmagnetoelasticfield_kernel<dataT>(Bx_acc, By_acc, Bz_acc,
						                           mx_acc, my_acc, mz_acc,
												   exx_acc, exx_mul,
												   eyy_acc, eyy_mul,
												   ezz_acc, ezz_mul,
												   exy_acc, exy_mul,
												   exz_acc, exz_mul,
												   eyz_acc, eyz_mul,
												   B1_acc, B1_mul,
												   B2_acc, B2_mul,
												   Ms_acc, Ms_mul,
												   N));
    });
}
