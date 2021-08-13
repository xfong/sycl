// adduniaxialanisotropy2 kernel

#include "include/amul.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
template <typename dataT>
inline void adduniaxialanisotropy2_fcn(sycl::nd_item<3> item,
                                       dataT*  BX, dataT*     BY, dataT*  BZ,
                                       dataT* mx_, dataT*    my_, dataT* mz_,
                                       dataT* Ms_, dataT  Ms_mul,
                                       dataT* k1_, dataT  k1_mul,
                                       dataT* k2_, dataT  k2_mul,
                                       dataT* ux_, dataT  ux_mul,
                                       dataT* uy_, dataT  uy_mul,
                                       dataT* uz_, dataT  uz_mul,
                                       size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        sycl::vec<dataT, 3>     u = normalized(vmul<dataT>(ux_, uy_, uz_, ux_mul, uy_mul, uz_mul, gid));
        dataT               invMs = inv_Msat<dataT>(Ms_, Ms_mul, gid);
        dataT                  K1 = amul<dataT>(k1_, k1_mul, gid);
        dataT                  K2 = amul<dataT>(k2_, k2_mul, gid);
        K1  *= invMs;
        K2  *= invMs;
        sycl::vec<dataT, 3>     m = {mx_[gid], my_[gid], mz_[gid]};
        dataT                  mu = sycl::dot(m, u);
        sycl::vec<dataT, 3>    Ba = (dataT)(2.0)*K1*    (mu)*u+
                                    (dataT)(4.0)*K2*pow3(mu)*u;

        BX[gid] += Ba.x();
        BY[gid] += Ba.y();
        BZ[gid] += Ba.z();
    }
}

// the function that launches the kernel
template <typename dataT>
void adduniaxialanisotropy2_t(dim3 blocks, dim3 threads, sycl::queue q,
                              dataT*  BX, dataT*     BY, dataT* BZ,
                              dataT*  mx, dataT*     my, dataT* mz,
                              dataT* Ms_, dataT  Ms_mul,
                              dataT* k1_, dataT  k1_mul,
                              dataT* k2_, dataT  k2_mul,
                              dataT* ux_, dataT  ux_mul,
                              dataT* uy_, dataT  uy_mul,
                              dataT* uz_, dataT  uz_mul,
                              size_t   N) {
    libMumax3clDeviceFcnCall(adduniaxialanisotropy2_fcn<dataT>, blocks, threads,
                              BX,     BY, BZ,
                              mx,     my, mz,
                             Ms_, Ms_mul,
                             k1_, k1_mul,
                             k2_, k2_mul,
                             ux_, ux_mul,
                             uy_, uy_mul,
                             uz_, uz_mul,
                               N);
}
