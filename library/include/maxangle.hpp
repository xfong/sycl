#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

// See maxangle.go for more details.
template <typename dataT>
class adddmibulk_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	    using uint8Acc =
		    sycl::accessor<uint8_t, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
		adddmibulk_kernel(write_accessor dstPtr,
		             read_accessor MxPtr, read_accessor MyPtr, read_accessor MzPtr,
					 read_accessor aLUT2dPtr,
					 uint8Acc regionsPtr,
					 size_t Nx, size_t Ny, size_t Nz,
					 uint8_t PBC)
		    :	dst(dstPtr),
				mx(MxPtr),
				my(MyPtr),
				mz(MzPtr),
				aLUT2d(aLUT2dPtr),
				regions(regionsPtr),
			    Nx(Nx), Ny(Ny), Nz(Nz),
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
			dataT angle  = (dataT)(0.0);

			int i_;    // neighbor index
			sycl::vec<dataT, 3> m_; // neighbor mag
			dataT a__; // inter-cell exchange stiffness

			// left neighbor
			i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);  // load m
			m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
			a__ = aLUT2d[symidx(r0, regions[i_])];
			if (a__ != 0) {
				angle = max(angle, acos(dot(m_,m0)));
			}

			// right neighbor
			i_  = idx(hclampx(ix+1), iy, iz);
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);
			m_  = ( is0(m_)? m0: m_ );
			a__ = aLUT2d[symidx(r0, regions[i_])];
			if (a__ != 0) {
				angle = max(angle, acos(dot(m_,m0)));
			}

			// back neighbor
			i_  = idx(ix, lclampy(iy-1), iz);
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);
			m_  = ( is0(m_)? m0: m_ );
			a__ = aLUT2d[symidx(r0, regions[i_])];
			if (a__ != 0) {
				angle = max(angle, acos(dot(m_,m0)));
			}

			// front neighbor
			i_  = idx(ix, hclampy(iy+1), iz);
			m_  = make_vec3(mx[i_], my[i_], mz[i_]);
			m_  = ( is0(m_)? m0: m_ );
			a__ = aLUT2d[symidx(r0, regions[i_])];
			if (a__ != 0) {
				angle = max(angle, acos(dot(m_,m0)));
			}

			// only take vertical derivative for 3D sim
			if (Nz != 1) {
				// bottom neighbor
				i_  = idx(ix, iy, lclampz(iz-1));
				m_  = make_vec3(mx[i_], my[i_], mz[i_]);
				m_  = ( is0(m_)? m0: m_ );
				a__ = aLUT2d[symidx(r0, regions[i_])];
				if (a__ != 0) {
					angle = max(angle, acos(dot(m_,m0)));
				}

				// top neighbor
				i_  = idx(ix, iy, hclampz(iz+1));
				m_  = make_vec3(mx[i_], my[i_], mz[i_]);
				m_  = ( is0(m_)? m0: m_ );
				a__ = aLUT2d[symidx(r0, regions[i_])];
				if (a__ != 0) {
					angle = max(angle, acos(dot(m_,m0)));
				}
			}

			dst[I] = angle;
		}
	private:
	    write_accessor dst;
	    read_accessor  mx;
	    read_accessor  my;
	    read_accessor  mz;
	    read_accessor  aLUT2d;
	    uint8Acc       regions;
		size_t         Nx;
		size_t         Ny;
		size_t         Nz;
		uint8_t        PBC;
};

template <typename dataT>
void adddmibulk_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 sycl::buffer<dataT, 1> *mx, sycl::buffer<dataT, 1> *my, sycl::buffer<dataT, 1> *mz,
				 sycl::buffer<dataT, 1> *aLUT2d,
				 sycl::buffer<uint8_t, 1> *regions,
                 size_t Nx, size_t Ny, size_t Nz,
				 uint8_t PBC,
				 size_t gsize[3],
				 size_t lsize[3]) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read_write>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read_write>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read_write>(cgh);
        auto aLUT2d_acc = aLUT2d->template get_access<sycl::access::mode::read_write>(cgh);
        auto regions_acc = regions->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(gsize[0], gsize[1], gsize(2)),
		                                   sycl::range<3>(lsize[0], lsize[1], lsize(2))),
		                 adddmibulk_kernel<dataT>(dst_acc,
											mx_acc, my_acc, mz_acc,
											aLUT2d_acc, regions,
											Nx, Ny, Nz,
											PBC));
    });
}
