// addslonczewskitorque2 kernel

#include "include/amul.hpp"
#include "include/constants.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016

template <typename dataT>
inline void addslonczewskitorque2_fcn(sycl::nd_item<3> item,
                                      dataT*        tx, dataT*           ty, dataT* tz,
                                      dataT*       mx_, dataT*          my_, dataT* mz_,
                                      dataT*       Ms_, dataT        Ms_mul,
                                      dataT*       jz_, dataT        jz_mul,
                                      dataT*       px_, dataT        px_mul,
                                      dataT*       py_, dataT        py_mul,
                                      dataT*       pz_, dataT        pz_mul,
                                      dataT*    alpha_, dataT     alpha_mul,
                                      dataT*      pol_, dataT       pol_mul,
                                      dataT*   lambda_, dataT    lambda_mul,
                                      dataT* epsPrime_, dataT  epsPrime_mul,
                                      dataT*      flt_, dataT       flt_mul,
                                      size_t         N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dataT  J  = amul<dataT>(jz_, jz_mul, gid);
        dataT  Ms = amul<dataT>(Ms_, Ms_mul, gid);
        if (J == (dataT)(0.0) || Ms == (dataT)(0.0)) {
            return;
        }

        sycl::vec<dataT, 3> m = make_vec3<dataT>(mx_[gid], my_[gid], mz_[gid]);
        sycl::vec<dataT, 3> p = normalized<dataT>(vmul<dataT>(px_, py_, pz_, px_mul, py_mul, pz_mul, gid));

        dataT  alpha        = amul<dataT>(alpha_, alpha_mul, gid);
        dataT  flt          = amul<dataT>(flt_, flt_mul, gid);
        dataT  pol          = amul<dataT>(pol_, pol_mul, gid);
        dataT  lambda       = amul<dataT>(lambda_, lambda_mul, gid);
        dataT  epsilonPrime = amul<dataT>(epsPrime_, epsPrime_mul, gid);

        dataT beta    = (HBAR / QE) * (J / (flt*Ms) );
        dataT lambda2 = lambda * lambda;
        dataT epsilon = pol * lambda2 / ((lambda2 + (dataT)(1.0)) + (lambda2 - (dataT)(1.0)) * sycl::dot(p, m));

        dataT A = beta * epsilon;
        dataT B = beta * epsilonPrime;

        dataT gilb     = (dataT)(1.0) / ((dataT)(1.0) + alpha * alpha);
        dataT mxpxmFac = gilb * (A + alpha * B);
        dataT pxmFac   = gilb * (B - alpha * A);

        sycl::vec<dataT, 3> pxm      = sycl::cross(p, m);
        sycl::vec<dataT, 3> mxpxm    = sycl::cross(m, pxm);

        tx[gid] += mxpxmFac * mxpxm.x() + pxmFac * pxm.x();
        ty[gid] += mxpxmFac * mxpxm.y() + pxmFac * pxm.y();
        tz[gid] += mxpxmFac * mxpxm.z() + pxmFac * pxm.z();
    }
}

template <typename dataT>
void addslonczewskitorque2_t(dim3 blocks, dim3 threads, sycl::queue q,
                             dataT*       tx, dataT*           ty, dataT* tz,
                             dataT*       mx, dataT*           my, dataT* mz,
                             dataT*       Ms, dataT        Ms_mul,
                             dataT*       jz, dataT        jz_mul,
                             dataT*       px, dataT        px_mul,
                             dataT*       py, dataT        py_mul,
                             dataT*       pz, dataT        pz_mul,
                             dataT*    alpha, dataT     alpha_mul,
                             dataT*      pol, dataT       pol_mul,
                             dataT*   lambda, dataT    lambda_mul,
                             dataT* epsPrime, dataT  epsPrime_mul,
                             dataT*      flt, dataT       flt_mul,
                             size_t        N) {
    libMumax3clDeviceFcnCall(addslonczewskitorque2_fcn<dataT>, blocks, threads,
                                   tx,           ty, tz,
                                   mx,           my, mz,
                                   Ms,       Ms_mul,
                                   jz,       jz_mul,
                                   px,       px_mul,
                                   py,       py_mul,
                                   pz,       pz_mul,
                                alpha,    alpha_mul,
                                  pol,      pol_mul,
                               lambda,   lambda_mul,
                             epsPrime, epsPrime_mul,
                                  flt,      flt_mul,
                                    N);
}
