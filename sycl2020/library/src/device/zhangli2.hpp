// addzhanglitorque2 kernel

#include "include/amul.hpp"

#include "include/constants.hpp"
#define PREFACTOR ((MUB) / (2 * QE * GAMMA0))

#include "include/stencil.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// spatial derivatives without dividing by cell size
#define deltax(in) (in[idx(hclampx(ix+1), iy, iz)] - in[idx(lclampx(ix-1), iy, iz)])
#define deltay(in) (in[idx(ix, hclampy(iy+1), iz)] - in[idx(ix, lclampy(iy-1), iz)])
#define deltaz(in) (in[idx(ix, iy, hclampz(iz+1))] - in[idx(ix, iy, lclampz(iz-1))])

// device side function. This is essentially the function of the kernel
template <typename dataT>
inline void addzhanglitorque2_fcn(sycl::nd_item<3> item,
                                  dataT*     TX, dataT*       TY, dataT* TZ,
                                  dataT*     mx, dataT*       my, dataT* mz,
                                  dataT*    Ms_, dataT    Ms_mul,
                                  dataT*    jx_, dataT    jx_mul,
                                  dataT*    jy_, dataT    jy_mul,
                                  dataT*    jz_, dataT    jz_mul,
                                  dataT* alpha_, dataT alpha_mul,
                                  dataT*    xi_, dataT    xi_mul,
                                  dataT*   pol_, dataT   pol_mul,
                                  dataT      cx, dataT        cy, dataT  cz,
                                  size_t     Nx, size_t       Ny, size_t Nz,
                                  uint8_t   PBC) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    int i = idx(ix, iy, iz);

    dataT alpha              = amul<dataT>(alpha_, alpha_mul, i);
    dataT xi                 = amul<dataT>(xi_, xi_mul, i);
    dataT pol                = amul<dataT>(pol_, pol_mul, i);
    dataT invMs              = inv_Msat<dataT>(Ms_, Ms_mul, i);
    dataT b                  = invMs * PREFACTOR / ((dataT)(1.0) + xi*xi);
    sycl::vec<dataT, 3> Jvec = vmul<dataT>(jx_, jy_, jz_, jx_mul, jy_mul, jz_mul, i);
    sycl::vec<dataT, 3> J    = pol*Jvec;

    sycl::vec<dataT, 3> hspin = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0)); // (u·∇)m
    if (J.x() != (dataT)(0.0)) {
        hspin += (b/cx)*J.x() * make_vec3<dataT>(deltax(mx), deltax(my), deltax(mz));
    }
    if (J.y() != (dataT)(0.0)) {
        hspin += (b/cy)*J.y() * make_vec3<dataT>(deltay(mx), deltay(my), deltay(mz));
    }
    if (J.z() != (dataT)(0.0)) {
        hspin += (b/cz)*J.z() * make_vec3<dataT>(deltaz(mx), deltaz(my), deltaz(mz));
    }

    sycl::vec<dataT, 3> m      = make_vec3<dataT>(mx[i], my[i], mz[i]);
    sycl::vec<dataT, 3> torque = ((dataT)(-1.0)/((dataT)(1.0) + alpha*alpha)) * (
                                 ((dataT)(1.0)+xi*alpha) * sycl::cross(m, sycl::cross(m, hspin))
                                 +(  xi-alpha) * sycl::cross(m, hspin)           );

    // write back, adding to torque
    TX[i] += torque.x();
    TY[i] += torque.y();
    TZ[i] += torque.z();
}

// the function that launches the kernel
template <typename dataT>
void addzhanglitorque2_t(dim3 blocks, dim3 threads, sycl::queue q,
                         dataT*     TX, dataT*       TY, dataT* TZ,
                         dataT*     mx, dataT*       my, dataT* mz,
                         dataT*    Ms_, dataT    Ms_mul,
                         dataT*    jx_, dataT    jx_mul,
                         dataT*    jy_, dataT    jy_mul,
                         dataT*    jz_, dataT    jz_mul,
                         dataT* alpha_, dataT alpha_mul,
                         dataT*    xi_, dataT    xi_mul,
                         dataT*   pol_, dataT   pol_mul,
                         dataT      cx, dataT        cy, dataT  cz,
                         size_t     Nx, size_t       Ny, size_t Nz,
                         uint8_t   PBC) {
    libMumax3clDeviceFcnCall(addzhanglitorque2_fcn<dataT>, blocks, threads,
                                 TX,        TY, TZ,
                                 mx,        my, mz,
                                Ms_,    Ms_mul,
                                jx_,    jx_mul,
                                jy_,    jy_mul,
                                jz_,    jz_mul,
                             alpha_, alpha_mul,
                                xi_,    xi_mul,
                               pol_,   pol_mul,
                                 cx,        cy, cz,
                                 Nx,        Ny, Nz,
                                PBC);
}
