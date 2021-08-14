// getmagnetoelasticforce kernel

#include "include/amul.hpp"
#include "include/constants.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016
template <typename dataT>
inline void getmagnetoelasticforce_fcn(sycl::nd_item<3> item,
                                       dataT*  fx, dataT*    fy, dataT* fz,
                                       dataT*  mx, dataT*    my, dataT* mz,
                                       dataT* B1_, dataT B1_mul,
                                       dataT* B2_, dataT B2_mul,
                                       dataT rcsx, dataT rcsy, dataT rcsz,
                                       size_t  Nx, size_t  Ny, size_t  Nz,
                                       uint8_t PBC) {
    size_t ix = syclBlockIdx_x * syclBlockDim_x + syclThreadIdx_x;
    size_t iy = syclBlockIdx_y * syclBlockDim_y + syclThreadIdx_y;
    size_t iz = syclBlockIdx_z * syclBlockDim_z + syclThreadIdx_z;

    if ((ix >= Nx) || (iy >= Ny) || (iz >= Nz)) {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    sycl::vec<dataT, 3> m0   = make_vec3<dataT>(mx[I], my[I], mz[I]);                         // +0
    sycl::vec<dataT, 3> dmdx = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  //  ^h^bm/ ^h^bx
    sycl::vec<dataT, 3> dmdy = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  //  ^h^bm/ ^h^by
    sycl::vec<dataT, 3> dmdz = make_vec3<dataT>((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));  //  ^h^bm/ ^h^bz
    int i_;                                       // neighbor index

    //  ^h^bm/ ^h^bx
    {
        sycl::vec<dataT, 3> m_m2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // -2
        i_ = idx(lclampx(ix-2), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-2 >= 0 || PBCx)
        {
            m_m2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_m1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // -1
        i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx)
        {
            m_m1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_p1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // +1
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx)
        {
            m_p1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_p2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));     // +2
        i_ = idx(hclampx(ix+2), iy, iz);
        if (ix+2 < Nx || PBCx)
        {
            m_p2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdx = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdx = (dataT)(0.5) * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,     ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdx =  m0 - m_m1;                                             // -11-- backward difference,    ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdx = -m0 + m_p1;                                             // --11- forward difference,     ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdx =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0;                 // 111-- backward difference,    ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdx = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0;                 // --111 forward difference,     ~ h^2
        }
        else
        {
            dmdx = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,     ~ h^4
        }
    }

    //  ^h^bm/ ^h^by
    {
        sycl::vec<dataT, 3> m_m2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, lclampy(iy-2), iz);
        if (iy-2 >= 0 || PBCy)
        {
            m_m2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_m1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, lclampy(iy-1), iz);
        if (iy-1 >= 0 || PBCy)
        {
            m_m1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_p1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, hclampy(iy+1), iz);
        if  (iy+1 < Ny || PBCy)
        {
             m_p1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_p2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, hclampy(iy+2), iz);
        if  (iy+2 < Ny || PBCy)
        {
             m_p2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdy = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdy = (dataT)(0.5) * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,     ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdy =  m0 - m_m1;                                             // -11-- backward difference,    ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdy = -m0 + m_p1;                                             // --11- forward difference,     ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdy =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0;                 // 111-- backward difference,    ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdy = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0;                 // --111 forward difference,     ~ h^2
        }
        else
        {
            dmdy = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,     ~ h^4
        }
    }


    //  ^h^bu/ ^h^bz
    {
        sycl::vec<dataT, 3> m_m2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, iy, lclampz(iz-2));
        if (iz-2 >= 0 || PBCz)
        {
            m_m2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_m1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, iy, lclampz(iz-1));
        if (iz-1 >= 0 || PBCz)
        {
            m_m1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_p1 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, iy, hclampz(iz+1));
        if  (iz+1 < Nz || PBCz)
        {
             m_p1 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        sycl::vec<dataT, 3> m_p2 = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));
        i_ = idx(ix, iy, hclampz(iz+2));
        if  (iz+2 < Nz || PBCz)
        {
             m_p2 = make_vec3(mx[i_], my[i_], mz[i_]);
        }

        if (is0(m_p1) && is0(m_m1))                                        //  +0
        {
            dmdz = make_vec3((dataT)(0.0), (dataT)(0.0), (dataT)(0.0));                          // --1-- zero
        }
        else if ((is0(m_m2) | is0(m_p2)) && !is0(m_p1) && !is0(m_m1))
        {
            dmdz = (dataT)(0.5) * (m_p1 - m_m1);                                   // -111-, 1111-, -1111 central difference,     ~ h^2
        }
        else if (is0(m_p1) && is0(m_m2))
        {
            dmdz =  m0 - m_m1;                                             // -11-- backward difference,    ~ h^1
        }
        else if (is0(m_m1) && is0(m_p2))
        {
            dmdz = -m0 + m_p1;                                             // --11- forward difference,     ~ h^1
        }
        else if (!is0(m_m2) && is0(m_p1))
        {
            dmdz =  (dataT)(0.5) * m_m2 - (dataT)(2.0) * m_m1 + (dataT)(1.5) * m0;                 // 111-- backward difference,    ~ h^2
        }
        else if (!is0(m_p2) && is0(m_m1))
        {
            dmdz = (dataT)(-0.5) * m_p2 + (dataT)(2.0) * m_p1 - (dataT)(1.5) * m0;                 // --111 forward difference,     ~ h^2
        }
        else
        {
            dmdz = (dataT)(2.0/3.0)*(m_p1 - m_m1) + (dataT)(1.0/12.0)*(m_m2 - m_p2); // 11111 central difference,     ~ h^4
        }
    }

    dmdx *= rcsx;
    dmdy *= rcsy;
    dmdz *= rcsz;

    dataT B1 = amul(B1_, B1_mul, I);
    dataT B2 = amul(B2_, B2_mul, I);

    fx[I] = (dataT)(2.0)*B1*m0.x()*dmdx.x() + B2*(m0.x()*(dmdy.y() + dmdz.z()) + m0.y()*dmdy.x() + m0.z()*dmdz.x());
    fy[I] = (dataT)(2.0)*B1*m0.y()*dmdy.y() + B2*(m0.x()*dmdx.y() + m0.y()*(dmdx.x() + dmdz.z()) + m0.z()*dmdz.y());
    fz[I] = (dataT)(2.0)*B1*m0.z()*dmdz.z() + B2*(m0.x()*dmdx.z() + m0.y()*dmdy.z() + m0.z()*(dmdx.x() + dmdy.y()));
}

// the function that launches the kernel
template <typename dataT>
void getmagnetoelasticforce_t(dim3 blocks, dim3 threads, sycl::queue q,
                              dataT*  fx, dataT*    fy, dataT*  fz,
                              dataT*  mx, dataT*    my, dataT*  mz,
                              dataT* B1_, dataT B1_mul,
                              dataT* B2_, dataT B2_mul,
                              dataT rcsx, dataT   rcsy, dataT rcsz,
                              size_t  Nx, size_t    Ny, size_t  Nz,
                              uint8_t PBC) {
    libMumax3clDeviceFcnCall(getmagnetoelasticforce_fcn<dataT>, blocks, threads,
                               fx,     fy,   fz,
                               mx,     my,   mz,
                              B1_, B1_mul,
                              B2_, B2_mul,
                             rcsx,   rcsy, rcsz,
                               Nx,     Ny,   Nz,
                              PBC);
}
