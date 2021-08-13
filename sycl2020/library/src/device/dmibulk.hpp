// adddmibulk kernel
#include "include/amul.hpp"
#include "include/exchange.hpp"
#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
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
inline void adddmibulk_fcn(sycl::nd_item<3> item,
                           dataT*        Hx, dataT*      Hy, dataT* Hz,
                           dataT*        mx, dataT*      my, dataT* mz,
                           dataT*       Ms_, dataT   Ms_mul,
                           dataT*    aLUT2d, dataT*  DLUT2d,
                           uint8_t* regions,
                           dataT         cx, dataT       cy, dataT  cz,
                           size_t        Nx, size_t      Ny, size_t Nz,
                           uint8_t      PBC, uint8_t OpenBC) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    int I = idx(ix, iy, iz);                                                             // central cell index
    sycl::vec<dataT, 3> h  = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0)); // add to H
    sycl::vec<dataT, 3> m0 = make_vec3<dataT>(mx[I], my[I], mz[I]);                      // central m
    uint8_t r0 = regions[I];
    int i_;                                                                              // neighbor index

    if(is0(m0)) {
        return;
    }

    // x derivatives (along length)
    {
        sycl::vec<dataT, 3> m1 = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // left neighbor
        i_ = idx(lclampx(ix-1), iy, iz);                                                  // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx) {
            m1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];
        dataT A = aLUT2d[symidx(r0, r1)];
        dataT D = DLUT2d[symidx(r0, r1)];
        dataT D_2A = D/((dataT)(2.0)*A);
        if (!is0(m1) || !OpenBC){                                                         // do nothing at an open boundary
            if (is0(m1)) {                                                                // neighbor missing
                m1 = {                         m0.x(),
                       m0.y() - (-cx * D_2A * m0.z()),
                       m0.z() + (-cx * D_2A * m0.y())};
            }
            h += ((dataT)(2.0)*A/(cx*cx)) * (m1 - m0);                                    // exchange
            h += {    (dataT)(0.0),
                  (D/cx)*(-m1.z()),
                  (D/cx)*( m1.y())};
        }
    }


    {
        sycl::vec<dataT, 3> m2 = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // right neighbor
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx) {
            m2 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m2)? r0 : regions[i_];
        dataT A = aLUT2d[symidx(r0, r1)];
        dataT D = DLUT2d[symidx(r0, r1)];
        dataT D_2A = D/((dataT)(2.0)*A);
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2 = {                         m0.x(),
                       m0.y() - (+cx * D_2A * m0.z()),
                       m0.z() + (+cx * D_2A * m0.y())};
            }
            h += ((dataT)(2.0)*A/(cx*cx)) * (m2 - m0);
            h += {     (dataT)(0.0),
                    (D/cx)*(m2.z()),
                  (D/cx)*(- m2.y())};
        }
    }

    // y derivatives (along height)
    {
        sycl::vec<dataT, 3> m1 = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy) {
            m1 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];
        dataT A = aLUT2d[symidx(r0, r1)];
        dataT D = DLUT2d[symidx(r0, r1)];
        dataT D_2A = D/((dataT)(2.0)*A);
        if (!is0(m1) || !OpenBC){
            if (is0(m1)) {
                m1 = { m0.x() + (-cy * D_2A * m0.z()),
                                               m0.y(),
                       m0.z() - (-cy * D_2A * m0.x())};
            }
            h += ((dataT)(2.0)*A/(cy*cy)) * (m1 - m0);
            h += { (D/cy)*(m1.z()),
                      (dataT)(0.0),
                  (D/cy)*(-m1.x())};
        }
    }

    {
        sycl::vec<dataT, 3> m2 = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy) {
            m2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m2)? r0 : regions[i_];
        dataT A = aLUT2d[symidx(r0, r1)];
        dataT D = DLUT2d[symidx(r0, r1)];
        dataT D_2A = D/((dataT)(2.0)*A);
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2 = { m0.x() + (+cy * D_2A * m0.z()),
                                               m0.y(),
                       m0.z() - (+cy * D_2A * m0.x())};
            }
            h += ((dataT)(2.0)*A/(cy*cy)) * (m2 - m0);
            h += { (D/cy)*(- m2.z()),
                        (dataT)(0.0),
                     (D/cy)*(m2.x())};
        }
    }

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        {
            sycl::vec<dataT, 3> m1 = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
            i_ = idx(ix, iy, lclampz(iz-1));
            if (iz-1 >= 0 || PBCz) {
                m1 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
            }
            int r1 = is0(m1)? r0 : regions[i_];
            dataT A = aLUT2d[symidx(r0, r1)];
            dataT D = DLUT2d[symidx(r0, r1)];
            dataT D_2A = D/((dataT)(2.0)*A);
            if (!is0(m1) || !OpenBC){
                if (is0(m1)) {
                    m1 = { m0.x() - (-cz * D_2A * m0.y()),
                           m0.y() + (-cz * D_2A * m0.x()),
                                                   m0.z()};
                }
                h += ((dataT)(2.0)*A/(cz*cz)) * (m1 - m0);
                h += { (D/cz)*(- m1.y()),
                       (D/cz)*(  m1.x()),
                            (dataT)(0.0)};
            }
        }

        // top neighbor
        {
            sycl::vec<dataT, 3> m2 = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
            i_ = idx(ix, iy, hclampz(iz+1));
            if (iz+1 < Nz || PBCz) {
                m2 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
            }
            int r1 = is0(m2)? r0 : regions[i_];
            dataT A = aLUT2d[symidx(r0, r1)];
            dataT D = DLUT2d[symidx(r0, r1)];
            dataT D_2A = D/((dataT)(2.0)*A);
            if (!is0(m2) || !OpenBC){
                if (is0(m2)) {
                    m2 = { m0.x() - (+cz * D_2A * m0.y()),
                           m0.y() + (+cz * D_2A * m0.x()),
                                                   m0.z()};
                }
                h += ((dataT)(2.0)*A/(cz*cz)) * (m2 - m0);
                h += { (D/cz)*(  m2.y()),
                       (D/cz)*(- m2.x()),
                            (dataT)(0.0)};
            }
        }
    }

    // write back, result is H + Hdmi + Hex
    dataT invMs = inv_Msat<dataT>(Ms_, Ms_mul, I);
    Hx[I] += h.x()*invMs;
    Hy[I] += h.y()*invMs;
    Hz[I] += h.z()*invMs;
}

// the function that launches the kernel
template <typename dataT>
void adddmibulk_t(dim3 blocks, dim3 threads, sycl::queue q,
                  dataT*        Hx, dataT*      Hy, dataT* Hz,
                  dataT*        mx, dataT*      my, dataT* mz,
                  dataT*      Msat, dataT   Ms_mul,
                  dataT*    aLUT2d, dataT*  DLUT2d,
                  uint8_t* regions,
                  dataT         cx, dataT       cy, dataT  cz,
                  size_t        Nx, size_t      Ny, size_t Nz,
                  uint8_t      PBC, uint8_t OpenBC) {
    libMumax3clDeviceFcnCall(adddmibulk_fcn<dataT>, blocks, threads,
                                  Hx,     Hy, Hz,
                                  mx,     my, mz,
                                Msat, Ms_mul,
                              aLUT2d, DLUT2d,
                             regions,
                                  cx,     cy, cz,
                                  Nx,     Ny, Nz,
                                 PBC, OpenBC);
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
