// addexchange kernel

#include "device_function.hpp"

// Additional includes
#include "amul.hpp"
#include "exchange.hpp"
#include "stencil.hpp"

// device side function.This is essentially the function of the kernel
// Add exchange field to Beff.
//      m: normalized magnetization
//      B: effective field in Tesla
//      Aex_red: Aex / (Msat * 1e18 m2)
template<typename dataT>
void addexchange_fcn(sycl::nd_item<3> item,
                     dataT* Bx, dataT* By, dataT* Bz,
                     const dataT* mx, const dataT* my, const dataT* mz,
                     dataT* Ms_, dataT Ms_mul,
                     const dataT* aLUT2d, const uint8_t* regions,
                     dataT wx, dataT wy, dataT wz,
                     size_t Nx, size_t Ny, size_t Nz,
                     uint8_t PBC) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    sycl::vec<dataT, 3> m0 = make_vec3<dataT>(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    uint8_t r0 = regions[I];
    sycl::vec<dataT, 3> B  = make_vec3<dataT>(0.0, 0.0, 0.0);

    int i_;    // neighbor index
    sycl::vec<dataT, 3> m_; // neighbor mag
    dataT a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);  // load m
    m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m0);

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
    m_  = ( is0(m_)? m0: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m0);

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);
    }

    dataT invMs = inv_Msat<dataT>(Ms_, Ms_mul, I);
    Bx[I] += B.x()*invMs;
    By[I] += B.y()*invMs;
    Bz[I] += B.z()*invMs;
}

// the function that launches the kernel
template <typename dataT>
void addexchange_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                   dataT* Bx, dataT* By, dataT* Bz,
                   dataT* mx, dataT* my, dataT* mz,
                   dataT* Ms, dataT Ms_mul,
                   dataT* aLUT2d,
                   uint8_t* regions,
                   dataT wx, dataT wy, dataT wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC) {

    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=] (sycl::nd_item<3> item){
        addexchange_fcn<dataT>(item,
                               Bx, By, Bz,
                               mx, my, mz,
                               Ms, Ms_mul,
                               aLUT2d,
                               regions,
                               wx, wy, wz,
                               Nx, Ny, Nz,
                               PBC);
    });
}
