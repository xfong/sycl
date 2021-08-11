// settemperature2 kernel

#include "include/amul.hpp"

// device side function. This is essentially the function of the kernel
// TODO: this could act on x,y,z, so that we need to call it only once.
template <typename dataT>
void settemperature2_fcn(size_t totalThreads, sycl::nd_item<1> item,
                         dataT* B,
                         dataT* noise, dataT kB2_VgammaDt,
                         dataT* Ms_, dataT Ms_mul,
                         dataT* temp_, dataT temp_mul,
                         dataT* alpha_, dataT alpha_mul,
                         size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
        dataT invMs = inv_Msat<dataT>(Ms_, Ms_mul, gid);
        dataT temp = amul<dataT>(temp_, temp_mul, gid);
        dataT alpha = amul<dataT>(alpha_, alpha_mul, gid);
        B[gid] = noise[gid] * sycl::srqt(kB2_VgammaDt * alpha * temp * invMs);
    }
}

// the function that launches the kernel
template <typename dataT>
void settemperature2_t(size_t blocks, size_t threads, sycl::queue q,
                       dataT* B,
                       dataT* noise,
                       dataT kB2_VgammaDt,
                       dataT* Ms, dataT Ms_mul,
                       dataT* temp, dataT temp_mul,
                       dataT* alpha, dataT alpha_mul,
                       size_t N) {
    size_t totalThreads = blocks * threads;
    libMumax3clDeviceFcnCall(settemperature2_fcn<dataT>, totalThreads, threads,
                             B,
                             noise, kB2_VgammaDt,
                             Ms_, Ms_mul,
                             temp_, temp_mul,
                             alpha_, alpha_mul,
                             N);
}
