#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// Exchange + Dzyaloshinskii-Moriya interaction for bulk material.
// Energy:
//
// 	E  = D M . rot(M)
//
// Effective field:
//
// 	Hx = 2A/Bs nabla²Mx + 2D/Bs dzMy - 2D/Bs dyMz
// 	Hy = 2A/Bs nabla²My + 2D/Bs dxMz - 2D/Bs dzMx
// 	Hz = 2A/Bs nabla²Mz + 2D/Bs dyMx - 2D/Bs dxMy
//
// Boundary conditions:
//
// 	        2A dxMx = 0
// 	 D Mz + 2A dxMy = 0
// 	-D My + 2A dxMz = 0
//
// 	-D Mz + 2A dyMx = 0
// 	        2A dyMy = 0
// 	 D Mx + 2A dyMz = 0
//
// 	 D My + 2A dzMx = 0
// 	-D Mx + 2A dzMy = 0
// 	        2A dzMz = 0
//
template <typename dataT>
class adddmibulk_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using uint8Acc =
		    sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		adddmibulk_kernel(write_accessor HxPtr, write_accessor HyPtr, write_accessor HzPtr,
		             read_accessor MxPtr, read_accessor MyPtr, read_accessor MzPtr,
		             read_accessor MsPtr, dataT Ms_mul,
					 read_accessor aLUT2dPtr, read_accessor DLUT2dPtr,
					 uint8Acc regionsPtr,
					 dataT cx, dataT cy, dataT cz,
					 size_t Nx, size_t Ny, size_t Nz,
					 uint8_t PBC, uint8_t OpenBC)
		    :	Hx(HxPtr),
				Hy(HyPtr),
				Hz(HzPtr),
				mx(MxPtr),
				my(MyPtr),
				mz(MzPtr),
				Ms_(MsPtr),
				Ms_mul(Ms_mul),
				aLUT2d(aLUT2dPtr),
				DLUT2d(DLUT2dPtr),
				regions(regionsPtr),
				cx(cx), cy(cy), cz(cz),
			    Nx(Nx), Ny(Ny), Nz(Nz),
				PBC(PBC), OpenPBC(OpenBC)	{}
		void operator()(sycl::nd_item<3> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
			size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

			if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
				return;
			}

			size_t I = idx(ix, iy, iz);                      // central cell index

			// add to H
			dataT hx_ = (dataT)(0.0);
			dataT hy_ = (dataT)(0.0);
			dataT hz_ = (dataT)(0.0);

			// central m
			dataT m0x = mx[I];
			dataT m0y = my[I];
			dataT m0z = mz[I];

			uint8_t r0 = regions[I];
			int i_;                                       // neighbor index

			bool ism00 = (m0x == (dataT)(0.0)) && (m0y == (dataT)(0.0)) && (m0z == (dataT)(0.0));
			if(ism00) {
				return;
			}

			// x derivatives (along length)
			{
				// left neighbor
				dataT m1x = (dataT)(0.0);
				dataT m1y = (dataT)(0.0);
				dataT m1z = (dataT)(0.0);

				i_ = idx(lclampx(ix-1), iy, iz);               // load neighbor m if inside grid, keep 0 otherwise
				if (ix-1 >= 0 || PBCx) {
					m1x = mx[i_];
					m1y = my[i_];
					m1z = mz[i_];
				}

				bool ism10 = ((m1x == (dataT)(0.0)) && (m1y == (dataT)(0.0)) && (m1z == (dataT)(0.0)));

				int r1 = (ism10) ? r0 : regions[i_];
				dataT A = aLUT2d[symidx(r0, r1)];
				dataT D = DLUT2d[symidx(r0, r1)];
				dataT D_2A = D/((dataT)(2.0)*A);
				if (!ism10 || !OpenBC){                          // do nothing at an open boundary
					if (ism10) {                                 // neighbor missing
						dataT inFactor = -cx * D_2A;
						m1x = m0x;
						m1y = m0y - (inFactor * m0z);
						m1z = m0z + (inFactor * m0y);
					}
					dataT prefactor = (dataT)(2.0)*A/(cx*cx);
					hx_ += prefactor * (m1x - m0x);          // exchange
					hy_ += prefactor * (m1y - m0y);          // exchange
					hz_ += prefactor * (m1z - m0z);          // exchange
					prefactor = D/cx;
					hy_ -= prefactor*(m1z);
					hz_ += prefactor*(m1y);
				}
			}

			{
				// right neighbor
				dataT m2x = (dataT)(0.0);
				dataT m2y = (dataT)(0.0);
				dataT m2z = (dataT)(0.0);

				i_ = idx(hclampx(ix+1), iy, iz);
				if (ix+1 < Nx || PBCx) {
					m2x = mx[i_];
					m2y = my[i_];
					m2z = mz[i_];
				}

				bool ism20 = ((m2x == (dataT)(0.0)) && (m2y == (dataT)(0.0)) && (m2z == (dataT)(0.0)));

				int r1 = (ism20) ? r0 : regions[i_];
				dataT A = aLUT2d[symidx(r0, r1)];
				dataT D = DLUT2d[symidx(r0, r1)];
				dataT D_2A = D/((dataT)(2.0)*A);
				if (!ism20 || !OpenBC){
					if (ism20) {
						dataT inFactor = cx * D_2A;
						m2x = m0x;
						m2y = m0y - (inFactor * m0z);
						m2z = m0z + (inFactor * m0y);
					}
					dataT prefactor = (dataT)(2.0)*A/(cx*cx);
					hx_   += prefactor * (m2x - m0x);
					hy_   += prefactor * (m2y - m0y);
					hz_   += prefactor * (m2z - m0z);
					prefactor = D/cx;
					hy_ += prefactor*(m2z);
					hz_ -= prefactor*(m2y);
				}
			}

			// y derivatives (along height)
			{
				dataT m1x = (dataT)(0.0);
				dataT m1y = (dataT)(0.0);
				dataT m1z = (dataT)(0.0);
				i_ = idx(ix, lclampy(iy-1), iz);
				if (iy-1 >= 0 || PBCy) {
					m1x = mx[i_];
					m1y = my[i_];
					m1z = mz[i_];
				}

				bool ism10 = ((m1x == (dataT)(0.0)) && (m1y == (dataT)(0.0)) && (m1z == (dataT)(0.0)));

				int r1 = (ism10) ? r0 : regions[i_];
				dataT A = aLUT2d[symidx(r0, r1)];
				dataT D = DLUT2d[symidx(r0, r1)];
				dataT D_2A = D/((dataT)(2.0)*A);
				if (!ism10 || !OpenBC){
					if (ism10) {
						dataT inFactor = -cy * D_2A;
						m1x = m0x + (inFactor * m0z);
						m1y = m0y;
						m1z = m0z - (inFactor * m0x);
					}
					dataT prefactor = (dataT)(2.0)*A/(cy*cy);
					hx_ +=  * (m1x - m0x);
					hy_ +=  * (m1y - m0y);
					hz_ +=  * (m1z - m0z);
					prefactor = D/cy;
					hx_ += prefactor*(m1z);
					hz_ -= prefactor*(m1x);
				}
			}

			{
				dataT m2x = (dataT)(0.0);
				dataT m2y = (dataT)(0.0);
				dataT m2z = (dataT)(0.0);
				i_ = idx(ix, hclampy(iy+1), iz);
				if  (iy+1 < Ny || PBCy) {
					m2x = mx[i_];
					m2y = my[i_];
					m2z = mz[i_];
				}

				bool ism20 = ((m2x == (dataT)(0.0)) && (m2y == (dataT)(0.0)) && (m2z == (dataT)(0.0)));

				int r1 = (ism20) ? r0 : regions[i_];
				dataT A = aLUT2d[symidx(r0, r1)];
				dataT D = dLUT2d[symidx(r0, r1)];
				dataT D_2A = D/((dataT)(2.0)*A);
				if (!is0m20 || !OpenBC){
					if (ism20) {
						dataT inFactor = cy * D_2A;
						m2x = m0x + (inFactor * m0z);
						m2y = m0y;
						m2z = m0z - (inFactor * m0x);
					}
					dataT prefactor = (dataT)(2.0)*A/(cy*cy);
					hx_   += prefactor * (m2x - m0x);
					hy_   += prefactor * (m2y - m0y);
					hz_   += prefactor * (m2z - m0z);
					prefactor = D/cy;
					hx_ -= prefactor*(m2z);
					hz_ += prefactor*(m2x);
				}
			}

			// only take vertical derivative for 3D sim
			if (Nz != 1) {
				// bottom neighbor
				{
					dataT m1x = (dataT)(0.0);
					dataT m1y = (dataT)(0.0);
					dataT m1z = (dataT)(0.0);

					i_  = idx(ix, iy, lclampz(iz-1));
					if (iz-1 >= 0 || PBCz) {
						dataT m1x = mx[i_];
						dataT m1y = my[i_];
						dataT m1z = mz[i_];
					}

					bool ism10 = ((m1x == (dataT)(0.0)) && (m1y == (dataT)(0.0)) && (m1z == (dataT)(0.0)));

					int r1 = (ism10) ? r0 : regions[i_];
					dataT A = aLUT2d[symidx(r0, r1)];
					dataT D = DLUT2d[symidx(r0, r1)];
					dataT D_2A = D/((dataT)(2.0)*A);

					if (!ism10 || !OpenBC){
						if (ism10) {
							dataT inFactor = -cz * D_2A;
							m1x = m0x - (inFactor * m0y);
							m1y = m0y + (inFactor * m0x);
							m1z = m0z;
						}
						dataT prefactor = (dataT)(2.0)*A/(cz*cz);
						hx_ += prefactor * (m1x - m0x);
						hy_ += prefactor * (m1y - m0y);
						hz_ += prefactor * (m1z - m0z);
						prefactor = D/cz;
						hx_ -= prefactor*(m1y);
						hy_ += prefactor*(m1x);
					}
				}

				// top neighbor
				{
					dataT m2x = (dataT)(0.0);
					dataT m2y = (dataT)(0.0);
					dataT m2z = (dataT)(0.0);
					i_  = idx(ix, iy, hclampz(iz+1));
					if (iz+1 < Nz || PBCz) {
						m2x = mx[i_];
						m2y = my[i_];
						m2z = mz[i_];
					}

					bool ism20 = ((m2x == (dataT)(0.0)) && (m2y == (dataT)(0.0)) && (m2z == (dataT)(0.0)));
					int r1 = (ism20) ? r0 : regions[i_];
					dataT A = aLUT2d[symidx(r0, r1)];
					dataT D = DLUT2d[symidx(r0, r1)];
					dataT D_2A = D/((dataT)(2.0)*A);

					if (!ism20 || !OpenBC){
						if (ism20) {
							dataT inFactor = cz * D_2A;
							m2x = m0x - (inFactor * m0y);
							m2y = m0y + (inFactor * m0x);
							m2z = m0z;
						}
						dataT prefactor = (dataT)(2.0)*A/(cz*cz);
						hx_ += prefactor * (m2x - m0x);
						hy_ += prefactor * (m2y - m0y);
						hz_ += prefactor * (m2z - m0z);
						prefactor = (D/cz);
						hx_ += prefactor*(m2y);
						hy_ -= prefactor*(m2x);
					}
				}
			}

			// write back, result is H + Hdmi + Hex
			dataT invMs = inv_Msat(Ms_, Ms_mul, I);
			Hx[I] += hx_*invMs;
			Hy[I] += hy_*invMs;
			Hz[I] += hz_*invMs;
		}
	private:
	    write_accessor Hx;
	    write_accessor Hy;
	    write_accessor Hz;
	    read_accessor  mx;
	    read_accessor  my;
	    read_accessor  mz;
	    read_accessor  Ms_;
		dataT          Ms_mul;
	    read_accessor  aLUT2d;
	    read_accessor  DLUT2d;
	    uint8Acc       regions;
		dataT          cx;
		dataT          cy;
		dataT          cz;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		uint8_t        PBC;
		uint8_t        OpenBC;
};

template <typename dataT>
void adddmibulk_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *Hx, sycl::buffer<dataT, 1> *Hy, sycl::buffer<dataT, 1> *Hz,
                 sycl::buffer<dataT, 1> *mx, sycl::buffer<dataT, 1> *my, sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *Msat, dataT Ms_mul,
				 sycl::buffer<dataT, 1> *aLUT2d, sycl::buffer<dataT, 1> *DLUT2d,
				 sycl::buffer<uint8_t, 1> *regions,
                 dataT cx, dataT cy, dataT cz,
                 size_t Nx, size_t Ny, size_t Nz,
				 uint8_t PBC, uint8_t OpenBC,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto Hx_acc = Hx->template get_access<sycl::access::mode::read_write>(cgh);
        auto Hy_acc = Hy->template get_access<sycl::access::mode::read_write>(cgh);
        auto Hz_acc = Hz->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read_write>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read_write>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read_write>(cgh);
        auto Ms_acc = Msat->template get_access<sycl::access::mode::read_write>(cgh);
        auto aLUT2d_acc = aLUT2d->template get_access<sycl::access::mode::read_write>(cgh);
        auto DLUT2d_acc = DLUT2d->template get_access<sycl::access::mode::read>(cgh);
        auto regions_acc = regions->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 adddmibulk_kernel<dataT>(Hx_acc, Hy_acc, Hz_acc,
											mx_acc, my_acc, mz_acc,
											Ms_acc, Ms_mul,
											aLUT2d_acc, DLUT2d_acc, regions,
											cx, cy, cz,
											Nx, Ny, Nz,
											PBC, OpenBC));
    });
}

// Note on boundary conditions.
//
// We need the derivative and laplacian of m in point A, but e.g. C lies out of the boundaries.
// We use the boundary condition in B (derivative of the magnetization) to extrapolate m to point C:
// 	m_C = m_A + (dm/dx)|_B * cellsize
//
// When point C is inside the boundary, we just use its actual value.
//
// Then we can take the central derivative in A:
// 	(dm/dx)|_A = (m_C - m_D) / (2*cellsize)
// And the laplacian:
// 	lapl(m)|_A = (m_C + m_D - 2*m_A) / (cellsize^2)
//
// All these operations should be second order as they involve only central derivatives.
//
//    ------------------------------------------------------------------ *
//   |                                                   |             C |
//   |                                                   |          **   |
//   |                                                   |        ***    |
//   |                                                   |     ***       |
//   |                                                   |   ***         |
//   |                                                   | ***           |
//   |                                                   B               |
//   |                                               *** |               |
//   |                                            ***    |               |
//   |                                         ****      |               |
//   |                                     ****          |               |
//   |                                  ****             |               |
//   |                              ** A                 |               |
//   |                         *****                     |               |
//   |                   ******                          |               |
//   |          *********                                |               |
//   |D ********                                         |               |
//   |                                                   |               |
//   +----------------+----------------+-----------------+---------------+
//  -1              -0.5               0               0.5               1
//                                 x