// addoommfslonczewski kernel

#include "include/amul.hpp"
#include "include/constants.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// Sloczewski torque as calculated in OOMMF
template <typename dataT>
inline void addoommfslonczewskitorque_fcn(sycl::nd_item<3> item,
                                          dataT*          tx, dataT*             ty, dataT*  tz,
                                          dataT*         mx_, dataT*            my_, dataT* mz_,
                                          dataT*         Ms_, dataT          Ms_mul,
                                          dataT*         jz_, dataT          jz_mul,
                                          dataT*         px_, dataT          px_mul,
                                          dataT*         py_, dataT          py_mul,
                                          dataT*         pz_, dataT          pz_mul,
                                          dataT*      alpha_, dataT       alpha_mul,
                                          dataT*       pfix_, dataT        pfix_mul,
                                          dataT*      pfree_, dataT       pfree_mul,
                                          dataT*  lambdafix_, dataT   lambdafix_mul,
                                          dataT* lambdafree_, dataT  lambdafree_mul,
                                          dataT*   epsPrime_, dataT    epsPrime_mul,
                                          dataT*        flt_, dataT         flt_mul,
                                          size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
         dataT   J = amul<dataT>(jz_, jz_mul, gid);
         dataT  Ms = amul<dataT>(Ms_, Ms_mul, gid);
         if ((J == (dataT)(0.0)) || (Ms == (dataT)(0.0))) {
             return;
         }

         sycl::vec<dataT, 3> m = make_vec3<dataT>(mx_[gid], my[gid], mz[gid]);
         sycl::vec<dataT, 3> p = normalized<dataT>(vmul(px_, py_, pz_, px_mul, py_mul, pz_mul, gid));
         dataT  alpha        = amul<dataT>(alpha_, alpha_mul, gid);
         dataT  flt          = amul<dataT>(flt_, flt_mul, gid);
         dataT  pfix         = amul<dataT>(pfix_, pfix_mul, gid);
         dataT  pfree        = amul<dataT>(pfree_, pfree_mul, gid);
         dataT  lambdafix    = amul<dataT>(lambdafix_, lambdafix_mul, gid);
         dataT  lambdafree   = amul<dataT>(lambdafree_, lambdafix_mul, gid);
         dataT  epsilonPrime = amul<dataT>(epsilonPrime_, epsilonPrime_mul, gid);

         dataT beta    = (HBAR / QE) * (J / ((dataT)(2.0) *flt*Ms) );
         dataT lambdafix2 = lambdafix * lambdafix;
         dataT lambdafree2 = lambdafree * lambdafree;
         dataT lambdafreePlus = sycl::sqrt(lambdafree2 + (dataT)(1.0));
         dataT lambdafixPlus = sycl::sqrt(lambdafix2 + (dataT)(1.0));
         dataT lambdafreeMinus = sycl::sqrt(lambdafree2 - (dataT)(1.0));
         dataT lambdafixMinus = sycl::sqrt(lambdafix2 - (dataT)(1.0));
         dataT plus_ratio = lambdafreePlus / lambdafixPlus;
         dataT minus_ratio = (dataT)(1.0);
         if (lambdafreeMinus > 0) {
             minus_ratio = lambdafixMinus / lambdafreeMinus;
         }
         // Compute q_plus and q_minus
         dataT plus_factor = pfix * lambdafix2 * plus_ratio;
         dataT minus_factor = pfree * lambdafree2 * minus_ratio;
         dataT q_plus = plus_factor + minus_factor;
         dataT q_minus = plus_factor - minus_factor;
         dataT lplus2 = lambdafreePlus * lambdafixPlus;
         dataT lminus2 = lambdafreeMinus * lambdafixMinus;
         dataT pdotm = sycl::dot(p, m);
         dataT A_plus = lplus2 + (lminus2 * pdotm);
         dataT A_minus = lplus2 - (lminus2 * pdotm);
         dataT epsilon = (q_plus / A_plus) - (q_minus / A_minus);

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

// the function that launches the kernel
template <typename dataT>
void addoommfslonczewskitorque_t(dim3 blocks, dim3 threads, sycl::queue q,
                                 dataT*         tx, dataT*             ty, dataT* tz,
                                 dataT*         mx, dataT*             my, dataT* mz,
                                 dataT*         Ms, dataT          Ms_mul,
                                 dataT*         jz, dataT          jz_mul,
                                 dataT*         px, dataT          px_mul,
                                 dataT*         py, dataT          py_mul,
                                 dataT*         pz, dataT          pz_mul,
                                 dataT*      alpha, dataT       alpha_mul,
                                 dataT*       pfix, dataT        pfix_mul,
                                 dataT*      pfree, dataT       pfree_mul,
                                 dataT*  lambdafix, dataT   lambdafix_mul,
                                 dataT* lambdafree, dataT  lambdafree_mul,
                                 dataT*   epsPrime, dataT    epsPrime_mul,
                                 dataT*        flt, dataT         flt_mul,
                                 size_t          N) {
    libMumax3clDeviceFcnCall(addslonczewskitorque_fcn<dataT>, blocks, threads,
                                     tx,             ty, tz,
                                     mx,             my, mz,
                                     Ms,         Ms_mul,
                                     jz,         jz_mul,
                                     px,         px_mul,
                                     py,         py_mul,
                                     pz,         pz_mul,
                                  alpha,      alpha_mul,
                                   pfix,       pfix_mul,
                                  pfree,      pfree_mul,
                              lambdafix,  lambdafix_mul,
                             lambdafree, lambdafree_mul,
                               epsPrime,   epsPrime_mul,
                                    flt,        flt_mul,
                                      N);
}
