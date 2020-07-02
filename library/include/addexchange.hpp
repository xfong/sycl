#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"
#include "exchange.hpp"
#include "stencil.hpp"

// Add exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: Aex / (Msat * 1e18 m2)
template <typename dataT>
class addexchange_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using uint8Acc =
		    sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		addexchange_kernel(write_accessor BxPtr, write_accessor ByPtr, write_accessor BzPtr,
		             read_accessor mxPtr, read_accessor myPtr, read_accessor mzPtr,
					 read_accessor MsPtr, dataT Ms_mul,
					 read_accessor aLUT2dPtr, uint8Acc regionsPtr,
					 dataT wx, dataT wy, dataT wz,
					 size_t Nx, size_t Ny, size_t Nz,
					 uint8_t PBC)
		    :	Bx(BxPtr),
				By(ByPtr),
				Bz(BzPtr),
				mx(mxPtr),
				my(myPtr),
				mz(mzPtr),
				Ms_(MsPtr),
				Ms_mul(Ms_mul),
				aLUT2d(aLUT2dPtr),
				regions(regionsPtr),
				wx(Nx),
				wy(Ny),
				wz(Nz),
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

			// central cell
			int I = idx(ix, iy, iz);
			sycl::vec<dataT, 3> m0 = make_vec3(mx[I], my[I], mz[I]);

			if (is0(m0)) {
				return;
			}

			uint8_t r0 = regions[I];
			sycl::vec<dataT, 3> B  = make_vec3(0.0, 0.0, 0.0);

			int i_;    // neighbor index
			sycl::vec<dataT, 3> m_; // neighbor mag
			dataT a__; // inter-cell exchange stiffness

			// left neighbor
			i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);  // load m
			m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B += wx * a__ *(m_ - m0);

			// right neighbor
			i_  = idx(hclampx(ix+1), iy, iz);
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);
			m_  = ( is0(m_)? m0: m_ );
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B += wx * a__ *(m_ - m0);

			// back neighbor
			i_  = idx(ix, lclampy(iy-1), iz);
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);
			m_  = ( is0(m_)? m0: m_ );
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B += wy * a__ *(m_ - m0);

			// front neighbor
			i_  = idx(ix, hclampy(iy+1), iz);
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);
			m_  = ( is0(m_)? m0: m_ );
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B += wy * a__ *(m_ - m0);

			// only take vertical derivative for 3D sim
			if (Nz != 1) {
				// bottom neighbor
				i_  = idx(ix, iy, lclampz(iz-1));
				m_  = make_vec3(mx[i_], my[i_], mz[i_]);
				m_  = ( is0(m_)? m0: m_ );
				a__ = aLUT2d[symidx(r0, regions[i_])];
				B += wz * a__ *(m_ - m0);

				// top neighbor
				i_  = idx(ix, iy, hclampz(iz+1));
				m_  = make_vec3(mx[i_], my[i_], mz[i_]);
				m_  = ( is0(m_)? m0: m_ );
				a__ = aLUT2d[symidx(r0, regions[i_])];
				B += wz * a__ *(m_ - m0);
			}

			dataT invMs = inv_Msat(Ms_, Ms_mul, I);
			Bx[I] += B.x()*invMs;
			By[I] += B.y()*invMs;
			Bz[I] += B.z()*invMs;
		}
	private:
	    write_accessor Bx;
	    write_accessor By;
	    write_accessor Bz;
	    read_accessor  mx;
	    read_accessor  my;
	    read_accessor  mz;
	    read_accessor  Ms_;
		dataT          Ms_mul;
	    read_accessor  aLUT2d;
		uint8Acc       regions;
		dataT          wx;
		dataT          wy;
		dataT          wz;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		uint8_t        PBC;
};

template <typename dataT>
void addexchange_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *Bx,
                 sycl::buffer<dataT, 1> *By,
                 sycl::buffer<dataT, 1> *Bz,
                 sycl::buffer<dataT, 1> *mx,
                 sycl::buffer<dataT, 1> *my,
                 sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *Ms,
                 dataT Ms_mul,
                 sycl::buffer<dataT, 1> *aLUT2d,
                 sycl::buffer<uint8_t, 1> *regions,
                 dataT wx,
                 dataT wy,
                 dataT wz,
                 size_t Nx,
                 size_t Ny,
                 size_t Nz,
				 uint8_t PBC,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto Bx_acc = Bx->template get_access<sycl::access::mode::read_write>(cgh);
        auto By_acc = By->template get_access<sycl::access::mode::read_write>(cgh);
        auto Bz_acc = Bz->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto Ms_acc = Ms->template get_access<sycl::access::mode::read>(cgh);
        auto aLUT2d_acc = aLUT2d->template get_access<sycl::access::mode::read>(cgh);
        auto regions_acc = regions->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize[2]),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize[2])),
		                 addexchange_kernel<dataT>(Bx_acc, By_acc, Bz_acc, mx_acc, my_acc, mz_acc, Ms_acc, Ms_mul, aLUT2d_acc, regions_acc, wx, wy, wz, Nx, Ny, Nz, PBC));
    });
}
