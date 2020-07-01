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

			//central cell
			size_t I = idx(ix, iy, iz);

			dataT m0x = mx[I];
			dataT m0y = my[I];
			dataT m0z = mz[I];

			if ((m0x == (dataT)(0.0)) && (m0y == (dataT)(0.0)) && (m0z == (dataT)(0.0))) {
				return;
			}

			uint8_t r0 = regions[I];

			dataT B_x = (dataT)(0.0);
			dataT B_y = (dataT)(0.0);
			dataT B_z = (dataT)(0.0);

			size_t i_; // neighbor index
			dataT mx_; // neighbor mag
			dataT my_; // neighbor mag
			dataT mz_; // neighbor mag
			dataT a__; // inter-cell exchange stiffness

			// left neighbor
			i_ = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC

			// load m
			mx_ = mx[i_];
			my_ = my[i_];
			mz_ = mz[i_];

			// replace missing non-boundary neighbor
			if ((mx_ == (dataT)(0.0)) && (my_ == (dataT)(0.0)) && (mz_ == (dataT)(0.0))) {
				mx_ = m0x;
				my_ = m0y;
				mz_ = m0z;
			}
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B_x += wx * a__ * (mx_ - m0x);
			B_y += wx * a__ * (my_ - m0y);
			B_z += wx * a__ * (mz_ - m0z);

			// right neighbor
			i_ = idx(hclampx(ix+1), iy, iz);

			// load m
			mx_ = mx[i_];
			my_ = my[i_];
			mz_ = mz[i_];

			// replace missing non-boundary neighbor
			if ((mx_ == (dataT)(0.0)) && (my_ == (dataT)(0.0)) && (mz_ == (dataT)(0.0))) {
				mx_ = m0x;
				my_ = m0y;
				mz_ = m0z;
			}
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B_x += wx * a__ * (mx_ - m0x);
			B_y += wx * a__ * (my_ - m0y);
			B_z += wx * a__ * (mz_ - m0z);

			// back neighbor
			i_ = idx(ix, lclampy(iy-1), iz); 

			mx_ = mx[i_];
			my_ = my[i_];
			mz_ = mz[i_];

			if ((mx_ == (dataT)(0.0)) && (my_ == (dataT)(0.0)) && (mz_ == (dataT)(0.0))) {
				mx_ = m0x;
				my_ = m0y;
				mz_ = m0z;
			}
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B_x += wy * a__ * (mx_ - m0x);
			B_y += wy * a__ * (my_ - m0y);
			B_z += wy * a__ * (mz_ - m0z);

			// front neighbor
			i_ = idx(ix, hclampy(iy+1), iz);

			// load m
			mx_ = mx[i_];
			my_ = my[i_];
			mz_ = mz[i_];

			// replace missing non-boundary neighbor
			if ((mx_ == (dataT)(0.0)) && (my_ == (dataT)(0.0)) && (mz_ == (dataT)(0.0))) {
				mx_ = m0x;
				my_ = m0y;
				mz_ = m0z;
			}
			a__ = aLUT2d[symidx(r0, regions[i_])];
			B_x += wy * a__ * (mx_ - m0x);
			B_y += wy * a__ * (my_ - m0y);
			B_z += wy * a__ * (mz_ - m0z);

			// only take vertical derivative for 3D sim
			if (Nz != 1) {
				// bottom neighbor
				i_ = idx(ix, iy, lclampz(iz-1)); 

				mx_ = mx[i_];
				my_ = my[i_];
				mz_ = mz[i_];

				if ((mx_ == (dataT)(0.0)) && (my_ == (dataT)(0.0)) && (mz_ == (dataT)(0.0))) {
					mx_ = m0x;
					my_ = m0y;
					mz_ = m0z;
				}
				a__ = aLUT2d[symidx(r0, regions[i_])];
				B_x += wz * a__ * (mx_ - m0x);
				B_y += wz * a__ * (my_ - m0y);
				B_z += wz * a__ * (mz_ - m0z);

				// top neighbor
				i_ = idx(ix, iy, hclampz(iz+1));

				// load m
				mx_ = mx[i_];
				my_ = my[i_];
				mz_ = mz[i_];

				// replace missing non-boundary neighbor
				if ((mx_ == (dataT)(0.0)) && (my_ == (dataT)(0.0)) && (mz_ == (dataT)(0.0))) {
					mx_ = m0x;
					my_ = m0y;
					mz_ = m0z;
				}
				a__ = aLUT2d[symidx(r0, regions[i_])];
				B_x += wz * a__ * (mx_ - m0x);
				B_y += wz * a__ * (my_ - m0y);
				B_z += wz * a__ * (mz_ - m0z);
			}

			dataT invMs = inv_Msat(Ms_, Ms_mul, I);

			Bx[I] += B_x*invMs;
			By[I] += B_y*invMs;
			Bz[I] += B_z*invMs;

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
