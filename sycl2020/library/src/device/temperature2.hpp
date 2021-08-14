// settemperature2 kernel

#include "include/amul.hpp"
#include "include/utils.h"
#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// TODO: this could act on x,y,z, so that we need to call it only once.
template <typename dataT>
inline void settemperature2_fcn(sycl::nd_item<3> item,
                                dataT* B,
                                dataT* noise, dataT kB2_VgammaDt,
                                dataT* Ms_, dataT Ms_mul,
                                dataT* temp_, dataT temp_mul,
                                dataT* alpha_, dataT alpha_mul,
                                size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += syclThreadCount) {
        dataT invMs = inv_Msat<dataT>(Ms_, Ms_mul, gid);
        dataT temp = amul<dataT>(temp_, temp_mul, gid);
        dataT alpha = amul<dataT>(alpha_, alpha_mul, gid);
        B[gid] = noise[gid] * sycl::sqrt(kB2_VgammaDt * alpha * temp * invMs);
    }
}

// the function that launches the kernel
template <typename dataT>
void settemperature2_t(dim3 blocks, dim3 threads, sycl::queue q,
                       dataT*            B,
                       dataT*        noise,
                       dataT  kB2_VgammaDt,
                       dataT*           Ms, dataT    Ms_mul,
                       dataT*         temp, dataT  temp_mul,
                       dataT*        alpha, dataT alpha_mul,
                       size_t            N) {
    libMumax3clDeviceFcnCall(settemperature2_fcn<dataT>, blocks, threads,
                                        B,
                                    noise,
                             kB2_VgammaDt,
                                       Ms,    Ms_mul,
                                     temp,  temp_mul,
                                    alpha, alpha_mul,
                                        N);
}
