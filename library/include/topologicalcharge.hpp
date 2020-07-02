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
class settopologicalcharge_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		settopologicalcharge_kernel(write_accessor sPtr,
		             read_accessor mxPtr, read_accessor myPtr, read_accessor mzPtr,
					 dataT icxcy,
					 size_t Nx, size_t Ny, size_t Nz,
					 uint8_t PBC)
		    :	s(sPtr),
				mx(mxPtr),
				my(myPtr),
				mz(mzPtr),
				icxcy(icxcy),
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
			
			sycl::vec<dataT, 3> m0 = make_vec3(mx[I], my[I], mz[I]); // +0
			sycl::vec<dataT, 3> dmdx = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  // ∂m/∂x
			sycl::vec<dataT, 3> dmdy = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  // ∂m/∂y
			sycl::vec<dataT, 3> dmdx_x_dmdy = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0)); // ∂m/∂x ❌ ∂m/∂y
			int i_;                                       // neighbor index

			if(is0(m0))
			{
				s[I] = (dataT)(0.0);
				return;
			}

			// x derivatives (along length)
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

				if (is0(m_p1) && is0(m_m1))                       //  +0
				{
					dmdx = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));         // --1-- zero
				}
				else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
				{
					dmdx = (dataT)(0.5) * (m_p1 - m_m1);                  // -111-, 1111-, -1111 central difference,  ε ~ h^2
				}
				else if (is0(m_p1) && is0(m_m2))
				{
					dmdx =  m0 - m_m1;                            // -11-- backward difference, ε ~ h^1
				}
				else if (is0(m_m1) && is0(m_p2))
				{
					dmdx = -m0 + m_p1;                            // --11- forward difference,  ε ~ h^1
				}
				else if (!is0(m_m2) && is0(m_p1))
				{
					dmdx =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0; // 111-- backward difference, ε ~ h^2
				}
				else if (!is0(m_p2) && is0(m_m1))
				{
					dmdx = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0; // --111 forward difference,  ε ~ h^2
				}
				else
				{
					dmdx = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,  ε ~ h^4
				}
			}

			// y derivatives (along height)
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
			dmdx_x_dmdy = sycl::cross(dmdx, dmdy);

			s[I] = icxcy * sycl::dot(m0, dmdx_x_dmdy);

			}
		}
	private:
	    write_accessor s;
	    read_accessor  mx;
	    read_accessor  my;
	    read_accessor  mz;
		dataT          icxcy;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		uint8_t        PBC;
};

template <typename dataT>
void settopologicalcharge_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *s,
                 sycl::buffer<dataT, 1> *mx, sycl::buffer<dataT, 1> *my, sycl::buffer<dataT, 1> *mz,
				 dataT icxcy,
                 size_t Nx,
                 size_t Ny,
                 size_t Nz,
				 uint8_t PBC,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto tx_acc = s->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 settopologicalcharge_kernel<dataT>(tx_acc, ty_acc, tz_acc,
						                           mx_acc, my_acc, mz_acc,
												   Ms_acc, Ms_mul,
												   jz_acc, jz_mul,
												   px_acc, px_mul,
												   py_acc, py_mul,
												   pz_acc, pz_mul,
												   alpha_acc, alpha_mul,
												   pol_acc, pol_mul,
												   lambda_acc, lambda_mul,
												   epsPrime_acc, epsPrime_mul,
												   flt_acc, flt_mul,
												   N));
    });
}
