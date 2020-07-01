#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// 3D micromagnetic kernel multiplication:
//
// |Mx|   |Kxx Kxy Kxz|   |Mx|
// |My| = |Kxy Kyy Kyz| * |My|
// |Mz|   |Kxz Kyz Kzz|   |Mz|
//
// ~kernel has mirror symmetry along Y and Z-axis,
// apart form first row,
// and is only stored (roughly) half:
//
// K11, K22, K02:
// xxxxx
// aaaaa
// bbbbb
// ....
// bbbbb
// aaaaa
//
// K12:
// xxxxx
// aaaaa
// bbbbb
// ...
// -bbbb
// -aaaa

template <typename dataT>
class kernmulrsymm2dz_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		kernmulrsymm2dz_kernel(write_accessor fftMxPtr, write_accessor fftMyPtr, write_accessor fftMzPtr,
		             read_accessor fftKxxPtr, read_accessor fftKyyPtr, read_accessor fftKzzPtr,
		             read_accessor fftKyzPtr, read_accessor fftKxzPtr, read_accessor fftKxyPtr,
					 size_t Nx, size_t Ny, size_t Nz)
		    :	fftMx(fftMxPtr),
				fftMy(fftMyPtr),
				fftMz(fftMzPtr),
				fftKxx(fftKxxPtr),
				fftKyy(fftKyyPtr),
				fftKzz(fftKzzPtr),
				fftKyz(fftKyzPtr),
				fftKxz(fftKxzPtr),
				fftKxy(fftKxyPtr),
			    Nx(Nx), Ny(Ny), Nz(Nz) {}
		void operator()(sycl::nd_item<3> item) {
			size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
			size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
			size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

			if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
				return;
			}

			// fetch (complex) FFT'ed magnetization
			size_t I = (iz*Ny + iy)*Nx + ix;
			size_t e = 2 * I;
			dataT reMx = fftMx[e  ];
			dataT imMx = fftMx[e+1];
			dataT reMy = fftMy[e  ];
			dataT imMy = fftMy[e+1];
			dataT reMz = fftMz[e  ];
			dataT imMz = fftMz[e+1];

			// fetch kernel

			// minus signs are added to some elements if
			// reconstructed from symmetry.
			dataT signYZ = (dataT)(1.0);
			dataT signXZ = (dataT)(1.0);
			dataT signXY = (dataT)(1.0);

			// use symmetry to fetch from redundant parts:
			// mirror index into first quadrant and set signs.
			if (iy > Ny/2) {
				iy = Ny-iy;
				signYZ = -signYZ;
				signXY = -signXY;
			}
			if (iz > Nz/2) {
				iz = Nz-iz;
				signYZ = -signYZ;
				signXZ = -signXZ;
			}

			// fetch kernel element from non-redundant part
			// and apply minus signs for mirrored parts.
			I = (iz*(Ny/2+1) + iy)*Nx + ix; // Ny/2+1: only half is stored
			dataT Kxx = fftKxx[I];
			dataT Kyy = fftKyy[I];
			dataT Kzz = fftKzz[I];
			dataT Kyz = fftKyz[I] * signYZ;
			dataT Kxz = fftKxz[I] * signXZ;
			dataT Kxy = fftKxy[I] * signXY;

			// m * K matrix multiplication, overwrite m with result.
			fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
			fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;
			fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
			fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;
			fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
			fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;
		}
	private:
	    write_accessor fftMx;
	    write_accessor fftMy;
	    write_accessor fftMz;
	    read_accessor  fftKxx;
	    read_accessor  fftKyy;
	    read_accessor  fftKzz;
	    read_accessor  fftKyz;
	    read_accessor  fftKxz;
	    read_accessor  fftKxy;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
};

template <typename dataT>
void kernmulrsymm2dz_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *fftMx, sycl::buffer<dataT, 1> *fftMy, sycl::buffer<dataT, 1> *fftMz,
                 sycl::buffer<dataT, 1> *fftKxx, sycl::buffer<dataT, 1> *fftKyy, sycl::buffer<dataT, 1> *fftKzz,
                 sycl::buffer<dataT, 1> *fftKyz, sycl::buffer<dataT, 1> *fftKxz, sycl::buffer<dataT, 1> *fftKxy,
                 size_t Nx,
                 size_t Ny,
                 size_t Nz,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto fftMz_acc = fftMz->template get_access<sycl::access::mode::read_write>(cgh);
        auto fftKzz_acc = fftKzz->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 kernmulrsymm2dz_kernel<dataT>(fftMx_acc, fftMy_acc, fftMz_acc,
											fftKxx_acc, fftKyy_acc, fftKzz_acc, fftKyz_acc, fftKxz_acc, fftKxy_acc,
											Nx, Ny, Nz));
    });
}
