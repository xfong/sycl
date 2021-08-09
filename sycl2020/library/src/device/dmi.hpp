// adddmi kernel
#include "include/amul.hpp"
#include "include/exchange.hpp"
#include "include/stencil.hpp"

// device side function.This is essentially the function of the kernel
// Exchange + Dzyaloshinskii-Moriya interaction according to
// Bagdanov and R    ler, PRL 87, 3, 2001. eq.8 (out-of-plane symmetry breaking).
// Taking into account proper boundary conditions.
// m: normalized magnetization
// H: effective field in Tesla
// D: dmi strength / Msat, in Tesla*m
// A: Aex/Msat
template <typename dataT>
void adddmi_fcn(sycl::nd_item<3> item,
                dataT* Hx, dataT* Hy, dataT* Hz,
                dataT* mx, dataT* my, dataT* mz,
                dataT*    Ms_, dataT  Ms_mul,
                dataT* aLUT2d, dataT* dLUT2d,
                uint8_t* regions,
                dataT  cx, dataT  cy, dataT  cz,
                size_t Nx, size_t Ny, size_t Nz,
                uint8_t PBC, uint8_t OpenBC) {
    size_t ix = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    size_t iy = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);
    size_t iz = item.get_group(2) * item.get_local_range(2) + item.get_local_id(2);

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    int I = idx(ix, iy, iz); // central cell index

    sycl::vec<dataT, 3> h  = make_vec3<dataT>(0.0, 0.0, 0.0);       // add to H
    sycl::vec<dataT, 3> m0 = make_vec3<dataT>(mx[I], my[I], mz[I]); // central m
    uint8_t r0 = regions[I];
    int i_;                                                         // neighbor index

    if(is0(m0)) {
        return;
    }

    // x derivatives (along length)
    {
        sycl::vec<dataT, 3> m1 = make_vec3<dataT>(0.0, 0.0, 0.0);   // left neighbor
        i_ = idx(lclampx(ix-1), iy, iz);                            // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx) {
            m1 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];                         // don't use inter region params if m1=0
        dataT A1 = aLUT2d[symidx(r0, r1)];                          // inter-region Aex
        dataT D1 = dLUT2d[symidx(r0, r1)];                          // inter-region Dex
        if (!is0(m1) || !OpenBC){                                   // do nothing at an open boundary
            if (is0(m1)) {                                          // neighbor missing
                // extrapolate missing m from Neumann BC's
                m1 = { m0.x() - (-cx * ((dataT)(0.5)*D1/A1) * m0.z()),
                                                               m0.y(),
                       m0.z() + (-cx * ((dataT)(0.5)*D1/A1) * m0.x())};

                h += ((dataT)(2.0)*A1/(cx*cx)) * (m1 - m0);         // exchange
                h += {(D1/cx)*(- m1.z()),
                            (dataT)(0.0),
                        (D1/cx)*(m1.x())};
            }
        }
    }

    {
        sycl::vec<dataT, 3> m2 = make_vec3<dataT>(0.0, 0.0, 0.0);          // right neighbor
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx) {
            m2 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        }
        int r2 = is0(m2)? r0 : regions[i_];
        dataT A2 = aLUT2d[symidx(r0, r2)];
        dataT D2 = dLUT2d[symidx(r0, r2)];
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2 = { m0.x() - (cx * ((dataT)(0.5)*D2/A2) * m0.z()),
                                                              m0.y(),
                       m0.z() + (cx * ((dataT)(0.5)*D2/A2) * m0.x())};
            }
            h += ((dataT)(2.0)*A2/(cx*cx)) * (m2 - m0);
            h += {  (D2/cx)*(m2.z()),
                        (dataT)(0.0),
                  (D2/cx)*(- m2.x())};
        }
    }

    // y derivatives (along height)
    {
        sycl::vec<dataT, 3> m1 = make_vec3<dataT>(0.0, 0.0, 0.0);
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy) {
            m1 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];
        dataT A1 = aLUT2d[symidx(r0, r1)];
        dataT D1 = dLUT2d[symidx(r0, r1)];
        if (!is0(m1) || !OpenBC){
            if (is0(m1)) {
                m1 = {                                         m0.x(),
                       m0.y() - (-cy * ((dataT)(0.5)*D1/A1) * m0.z()),
                       m0.z() + (-cy * ((dataT)(0.5)*D1/A1) * m0.y())};
            }
            h += ((dataT)(2.0)*A1/(cy*cy)) * (m1 - m0);
            h += {      (dataT)(0.0),
                  (D1/cy)*(- m1.z()),
                  (D1/cy)*(  m1.y())};
        }
    }

    {
        sycl::vec<dataT, 3> m2 = make_vec3<dataT>(0.0, 0.0, 0.0);
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy) {
            m2 = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
        }
        int r2 = is0(m2)? r0 : regions[i_];
        dataT A2 = aLUT2d[symidx(r0, r2)];
        dataT D2 = dLUT2d[symidx(r0, r2)];
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2 = {                                        m0.x(),
                       m0.y() - (cy * ((dataT)(0.5)*D2/A2) * m0.z()),
                       m0.z() + (cy * ((dataT)(0.5)*D2/A2) * m0.y())};
            }
            h += ((dataT)(2.0)*A2/(cy*cy)) * (m2 - m0);
            h += {     (dataT)(0.0),
                   (D2/cy)*(m2.z()),
                  (D2/cy)*(-m2.y())};
        }
    }

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        {
            i_  = idx(ix, iy, lclampz(iz-1));
            sycl::vec<dataT, 3> m1  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
            m1  = ( is0(m1)? m0: m1 );                                 // Neumann BC
            dataT A1 = aLUT2d[symidx(r0, regions[i_])];
            h += ((dataT)(2.0)*A1/(cz*cz)) * (m1 - m0);                // Exchange only
        }

        // top neighbor
        {
            i_  = idx(ix, iy, hclampz(iz+1));
            sycl::vec<dataT, 3> m2  = make_vec3<dataT>(mx[i_], my[i_], mz[i_]);
            m2  = ( is0(m2)? m0: m2 );
            dataT A2 = aLUT2d[symidx(r0, regions[i_])];
            h += ((dataT)(2.0)*A2/(cz*cz)) * (m2 - m0);
        }
    }

    // write back, result is H + Hdmi + Hex
    dataT invMs = inv_Msat(Ms_, Ms_mul, I);
    Hx[I] += h.x()*invMs;
    Hy[I] += h.y()*invMs;
    Hz[I] += h.z()*invMs;
}

// the function that launches the kernel
template<typename dataT>
void adddmi_t(size_t blocks[3], size_t threads[3], sycl::queue q,
              dataT* Hx, dataT* Hy, dataT* Hz,
              dataT* mx, dataT* my, dataT* mz,
              dataT* Ms_, dataT Ms_mul,
              dataT* aLUT2d, dataT* dLUT2d,
              uint8_t* regions,
              size_t cx, size_t cy, size_t cz,
              size_t Nx, size_t Ny, size_t Nz,
              uint8_t PBC, uint8_t OpenBC) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item){
        adddmi_fcn<dataT>(item,
                          Hx, Hy, Hz,
                          mx, my, mz,
                          Ms_, Ms_mul,
                          aLUT2d, dLUT2d,
                          regions,
                          cx, cy, cz,
                          Nx, Ny, Nz,
                          PBC, OpenBC);
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
