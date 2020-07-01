#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"
#include "exchange.hpp"
#include "stencil.hpp"

// Finds the average exchange strength around each cell, for debugging.
template <typename dataT>
class exchangedecode_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using uint8Acc =
		    sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		exchangedecode_kernel(write_accessor dstPtr
					 read_accessor aLUT2dPtr, uint8Acc regionsPtr,
					 dataT wx, dataT wy, dataT wz,
					 size_t Nx, size_t Ny, size_t Nz,
					 uint8_t PBC)
		    :	dst(dstPtr),
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
			uint8_t r0 = regions[I];

			size_t i_; // neighbor index
			dataT avg = (dataT)(0.0);

			// left neighbor
			i_ = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
			avg += aLUT2d[symidx(r0, regions[i_])];

			// right neighbor
			i_ = idx(hclampx(ix+1), iy, iz);
			avg = aLUT2d[symidx(r0, regions[i_])];

			// back neighbor
			i_ = idx(ix, lclampy(iy-1), iz); 
			avg = aLUT2d[symidx(r0, regions[i_])];

			// front neighbor
			i_ = idx(ix, hclampy(iy+1), iz);
			avg = aLUT2d[symidx(r0, regions[i_])];

			// only take vertical derivative for 3D sim
			if (Nz != 1) {
				// bottom neighbor
				i_ = idx(ix, iy, lclampz(iz-1)); 
				avg = aLUT2d[symidx(r0, regions[i_])];

				// top neighbor
				i_ = idx(ix, iy, hclampz(iz+1));
				avg = aLUT2d[symidx(r0, regions[i_])];
			}

			dst[I] = avg;

		}
	private:
	    write_accessor dst;
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
void exchangedecode_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
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
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto aLUT2d_acc = aLUT2d->template get_access<sycl::access::mode::read>(cgh);
        auto regions_acc = regions->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize[2]),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize[2])),
		                 exchangedecode_kernel<dataT>(dst_acc, aLUT2d_acc, regions_acc, wx, wy, wz, Nx, Ny, Nz, PBC));
    });
}
