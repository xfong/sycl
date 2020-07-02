#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"
#include "constants.hpp"

// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016

template <typename dataT>
class getmagnetoelasticforce_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		getmagnetoelasticforce_kernel(write_accessor fxPtr, write_accessor fyPtr, write_accessor fzPtr,
		             read_accessor mxPtr, read_accessor myPtr, read_accessor mzPtr,
		             read_accessor B1Ptr, dataT B1_mul,
					 read_accessor B2Ptr, dataT B2_mul,
					 dataT rcsx, dataT rcsy, dataT rcsz,
					 size_t Nx, size_t, Ny, size_t Nz,
					 uint8_t PBC)
		    :	fx(fxPtr),
				fy(fyPtr),
				fz(fzPtr),
				mx(mxPtr),
				my(myPtr),
				mz(mzPtr),
				B1_(B1Ptr),
				B1_mul(B1_mul),
				B2_(B2Ptr),
				B2_mul(B2_mul),
				rcsx(rcsx),
				rcsy(rcsy),
				rcsz(rcsz),
				Nx(Nx),
				Ny(Ny),
				Nz(Nz),
				PBC(PBC) {}
		void operator()(sycl::nd_item<1> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
			size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

			if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
				return;
			}

			int I = idx(ix, iy, iz);                      // central cell index
			sycl::vec<dataT, 3> m0 = make_vec3(mx[I], my[I], mz[I]);                         // +0
			sycl::vec<dataT, 3> dmdx = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  // ∂m/∂x
			sycl::vec<dataT, 3> dmdy = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  // ∂m/∂y
			sycl::vec<dataT, 3> dmdz = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  // ∂m/∂z
			int i_;                                       // neighbor index

			// ∂m/∂x
			{	
				sycl::vec<dataT, 3> m_m2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // -2
				i_ = idx(lclampx(ix-2), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
				if (ix-2 >= 0 || PBCx)
				{
					m_m2 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_m1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // -1
				i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
				if (ix-1 >= 0 || PBCx)
				{
					m_m1 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_p1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // +1
				i_ = idx(hclampx(ix+1), iy, iz);
				if (ix+1 < Nx || PBCx)
				{
					m_p1 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_p2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // +2
				i_ = idx(hclampx(ix+2), iy, iz);
				if (ix+2 < Nx || PBCx)
				{
					m_p2 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				if (is0(m_p1) && is0(m_m1))                                        //  +0
				{
					dmdx = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));                          // --1-- zero
				}
				else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
				{
					dmdx = (dataT)(0.5) * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
				}
				else if (is0(m_p1) && is0(m_m2))
				{
					dmdx =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
				}
				else if (is0(m_m1) && is0(m_p2))
				{
					dmdx = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
				}
				else if (!is0(m_m2) && is0(m_p1))
				{
					dmdx =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0;                 // 111-- backward difference, ε ~ h^2
				}
				else if (!is0(m_p2) && is0(m_m1))
				{
					dmdx = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0;                 // --111 forward difference,  ε ~ h^2
				}
				else
				{
					dmdx = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
				}
			}

			// ∂m/∂y
			{
				sycl::vec<dataT, 3> m_m2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, lclampy(iy-2), iz);
				if (iy-2 >= 0 || PBCy)
				{
					m_m2 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_m1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, lclampy(iy-1), iz);
				if (iy-1 >= 0 || PBCy)
				{
					m_m1 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_p1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, hclampy(iy+1), iz);
				if  (iy+1 < Ny || PBCy)
				{
					m_p1 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_p2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, hclampy(iy+2), iz);
				if  (iy+2 < Ny || PBCy)
				{
					m_p2 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				if (is0(m_p1) && is0(m_m1))                                        //  +0
				{
					dmdy = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));                          // --1-- zero
				}
				else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
				{
					dmdy = (dataT)(0.5) * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
				}
				else if (is0(m_p1) && is0(m_m2))
				{
					dmdy =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
				}
				else if (is0(m_m1) && is0(m_p2))
				{
					dmdy = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
				}
				else if (!is0(m_m2) && is0(m_p1))
				{
					dmdy =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0;                 // 111-- backward difference, ε ~ h^2
				}
				else if (!is0(m_p2) && is0(m_m1))
				{
					dmdy = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0;                 // --111 forward difference,  ε ~ h^2
				}
				else
				{
					dmdy = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
				}
			}


			// ∂u/∂z
			{
				sycl::vec<dataT, 3> m_m2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, iy, lclampz(iz-2));
				if (iz-2 >= 0 || PBCz)
				{
					m_m2 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_m1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, iy, lclampz(iz-1));
				if (iz-1 >= 0 || PBCz)
				{
					m_m1 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_p1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, iy, hclampz(iz+1));
				if  (iz+1 < Nz || PBCz)
				{
					m_p1 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				sycl::vec<dataT, 3> m_p2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
				i_ = idx(ix, iy, hclampz(iz+2));
				if  (iz+2 < Nz || PBCz)
				{
					m_p2 = make_vec3(mx[i_], my[i_], mz[i_]);
				}

				if (is0(m_p1) && is0(m_m1))                                        //  +0
				{
					dmdz = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));                          // --1-- zero
				}
				else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
				{
					dmdz = (dataT)(0.5) * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,  ε ~ h^2
				}
				else if (is0(m_p1) && is0(m_m2))
				{
					dmdz =  m0 - m_m1;                                             // -11-- backward difference, ε ~ h^1
				}
				else if (is0(m_m1) && is0(m_p2))
				{
					dmdz = -m0 + m_p1;                                             // --11- forward difference,  ε ~ h^1
				}
				else if (!is0(m_m2) && is0(m_p1))
				{
					dmdz =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0;                 // 111-- backward difference, ε ~ h^2
				}
				else if (!is0(m_p2) && is0(m_m1))
				{
					dmdz = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0;                 // --111 forward difference,  ε ~ h^2
				}
				else
				{
					dmdz = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
				}
			}

			dmdx *= rcsx;
			dmdy *= rcsy;
			dmdz *= rcsz;

			dataT B1 = amul(B1_, B1_mul, I);
			dataT B2 = amul(B2_, B2_mul, I);

			fx[I] = (dataT)(2.0)*B1*m0.x()*dmdx.x() + B2*(m0.x()*(dmdy.y() + dmdz.z()) + m0.y()*dmdy.x() + m0.z()*dmdz.x());
			fy[I] = (dataT)(2.0)*B1*m0.y()*dmdy.y() + B2*(m0.x()*dmdx.y() + m0.y()*(dmdx.x() + dmdz.z()) + m0.z()*dmdz.y());
			fz[I] = (dataT)(2.0)*B1*m0.z()*dmdz.z() + B2*(m0.x()*dmdx.z() + m0.y()*dmdy.z() + m0.z()*(dmdx.x() + dmdy.y()));
			}
		}
	private:
	    write_accessor fx;
	    write_accessor fy;
	    write_accessor fz;
	    read_accessor  mx;
	    read_accessor  my;
	    read_accessor  mz;
	    read_accessor  B1_;
		dataT          B1_mul;
	    read_accessor  B2_;
		dataT          B2_mul;
		dataT          rcsx;
		dataT          rcsy;
		dataT          rcsz;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		uint8_t        PBC;
};

template <typename dataT>
void getmagnetoelasticforce_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *fx, sycl::buffer<dataT, 1> *fy, sycl::buffer<dataT, 1> *fz,
                 sycl::buffer<dataT, 1> *mx, sycl::buffer<dataT, 1> *my, sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *B1, dataT B1_mul,
                 sycl::buffer<dataT, 1> *B2, dataT B2_mul,
				 dataT rcsx, dataT rcsy, dataT rcsz,
                 size_t Nx, size_t Ny, size_t Nz,
				 uint8_t PBC,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto fx_acc = fx->template get_access<sycl::access::mode::read_write>(cgh);
        auto fy_acc = fy->template get_access<sycl::access::mode::read_write>(cgh);
        auto fz_acc = fz->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto B1_acc = B1->template get_access<sycl::access::mode::read>(cgh);
        auto B2_acc = B2->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 getmagnetoelasticforce_kernel<dataT>(fx_acc, fy_acc, fz_acc,
						                           mx_acc, my_acc, mz_acc,
												   B1_acc, B1_mul,
												   B2_acc, B2_mul,
												   rcsx, rcsy, rcsz,
												   Nx, Ny, Nz,
												   PBC));
    });
}
