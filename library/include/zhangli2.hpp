#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"

#include "constants.hpp"
#define PREFACTOR ((MUB) / (2 * QE * GAMMA0))

#include "stencil.hpp"

// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

template <typename dataT>
class addzhanglitorque2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		addzhanglitorque2_kernel(write_accessor TXPtr,
					 write_accessor TYPtr,
					 write_accessor TZPtr,
		             read_accessor mxPtr,
					 read_accessor myPtr,
					 read_accessor mzPtr,
		             read_accessor Ms_, dataT Ms_mul,
		             read_accessor jxPtr, dataT jx_mul,
					 read_accessor jyPtr, dataT jy_mul,
					 read_accessor jzPtr, dataT jz_mul,
		             read_accessor alpha_, dataT alpha_mul,
		             read_accessor xi_, dataT xi_mul,
		             read_accessor pol_, dataT pol_mul,
		             dataT cx, dataT cy, dataT cz,
					 size_t Nx, size_t Ny, size_t Nz,
					 uint8_t PBC)
		    :	TXPtr(TXPtr),
				TYPtr(TYPtr),
				TZPtr(TZPtr),
				mxPtr(mxPtr),
				myPtr(myPtr),
				mzPtr(mzPtr),
				Ms_(Ms_),
				Ms_mul(Ms_mul),
				jxPtr(jxPtr),
				jx_mul(jx_mul),
				jyPtr(jyPtr),
				jy_mul(jy_mul),
				jzPtr(jzPtr),
				jz_mul(jz_mul),
				alpha_(alpha_),
				alpha_mul(alpha_mul),
				xi_(xi_),
				xi_mul(xi_mul),
				pol_(pol_),
				pol_mul(pol_mul),
				cx(cx),
				cy(cy),
				cz(cz),
				Nx(Nx),
				Ny(Ny),
				Nz(Nz),
				PBC(PBC) {}
		void operator()(sycl::nd_item<3> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
			size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

			if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
				return;
			}

			int i = idx(ix, iy, iz);

			dataT alpha = amul(alpha_, alpha_mul, i);
			dataT xi    = amul(xi_, xi_mul, i);
			dataT pol   = amul(pol_, pol_mul, i);
			dataT invMs = inv_Msat(Ms_, Ms_mul, i);
			dataT b = invMs * PREFACTOR / ((dataT)(1.0) + xi*xi);
			sycl::vec<dataT, 3> Jvec = vmul(jxPtr, jyPtr, jzPtr, jx_mul, jy_mul, jz_mul, i);
			sycl::vec<dataT, 3> J = pol*Jvec;

			sycl::vec<dataT, 3> hspin = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0)); // (u·∇)m
			if (J.x() != (dataT)(0.0)) {
				hspin += (b/cx)*J.x() * make_vec3(deltax(mxPtr), deltax(myPtr), deltax(mzPtr));
			}
			if (J.y() != (dataT)(0.0)) {
				hspin += (b/cy)*J.y() * make_vec3(deltay(mxPtr), deltay(myPtr), deltay(mzPtr));
			}
			if (J.z() != (dataT)(0.0)) {
				hspin += (b/cz)*J.z() * make_vec3(deltaz(mxPtr), deltaz(myPtr), deltaz(mzPtr));
			}

			sycl::vec<dataT, 3> m      = make_vec3(mxPtr[i], myPtr[i], mzPtr[i]);
			sycl::vec<dataT, 3> torque = ((dataT)(-1.0)/((dataT)(1.0) + alpha*alpha)) * (
								((dataT)(1.0)+xi*alpha) * sycl::cross(m, sycl::cross(m, hspin))
								+(  xi-alpha) * sycl::cross(m, hspin)           );

			// write back, adding to torque
			TXPtr[i] += torque.x();
			TYPtr[i] += torque.y();
			TZPtr[i] += torque.z();
		}
	private:
	    write_accessor TXPtr;
	    write_accessor TYPtr;
	    write_accessor TZPtr;
	    read_accessor  mxPtr;
	    read_accessor  myPtr;
	    read_accessor  mzPtr;
	    read_accessor  Ms_;
		dataT          Ms_mul;
	    read_accessor  jxPtr;
		dataT          jx_mul;
	    read_accessor  jyPtr;
		dataT          jy_mul;
	    read_accessor  jzPtr;
		dataT          jz_mul;
	    read_accessor  alpha_;
		dataT          alpha_mul;
	    read_accessor  xi_;
		dataT          xi_mul;
	    read_accessor  pol_;
		dataT          pol_mul;
		dataT          cx;
		dataT          cy;
		dataT          cz;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		uint8_t        PBC;
};

template <typename dataT>
void addzhanglitorque2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *TX,
                 sycl::buffer<dataT, 1> *TY,
                 sycl::buffer<dataT, 1> *TZ,
                 sycl::buffer<dataT, 1> *mx,
                 sycl::buffer<dataT, 1> *my,
                 sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *Ms_, dataT Ms_mul,
                 sycl::buffer<dataT, 1> *jx_, dataT jx_mul,
                 sycl::buffer<dataT, 1> *jy_, dataT jy_mul,
                 sycl::buffer<dataT, 1> *jz_, dataT jz_mul,
                 sycl::buffer<dataT, 1> *alpha_, dataT alpha_mul,
                 sycl::buffer<dataT, 1> *xi_, dataT xi_mul,
				 sycl::buffer<dataT, 1> *pol_, dataT pol_mul,
				 dataT cx, dataT cy, dataT cz,
                 size_t Nx, size_t Ny, size_t Nz,
				 uint8_t PBC,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto TX_acc = TX->template get_access<sycl::access::mode::read_write>(cgh);
        auto TY_acc = TY->template get_access<sycl::access::mode::read_write>(cgh);
        auto TZ_acc = TZ->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto Ms_acc = Ms_->template get_access<sycl::access::mode::read>(cgh);
        auto jx_acc = jx_->template get_access<sycl::access::mode::read>(cgh);
        auto jy_acc = jy_->template get_access<sycl::access::mode::read>(cgh);
        auto jz_acc = jz_->template get_access<sycl::access::mode::read>(cgh);
        auto alpha_acc = alpha_->template get_access<sycl::access::mode::read>(cgh);
        auto xi_acc = xi_->template get_access<sycl::access::mode::read>(cgh);
        auto pol_acc = pol_->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<3>(gsize[0], gsize[1], gsize[2]),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize[2])),
		                 addzhanglitorque2_kernel<dataT>(TX_acc, TY_acc, TZ_acc,
														   mx_acc, my_acc, mz_acc,
						                                   Ms_acc, Ms_mul,
														   jx_acc, jx_mul,
														   jy_acc, jy_mul,
														   jz_acc, jz_mul,
														   alpha_acc, alpha_mul,
														   xi_acc, xi_mul,
														   pol_acc, pol_mul,
														   cx, cy, cz,
														   Nx, Ny, Nz,
														   PBC));
    });
}
