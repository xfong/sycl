// cubicanisotropy2 kernel

#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
#include "include/amul.hpp"

template <typename dataT>
void addcubicanisotropy2_fcn(size_t totalThreads, sycl::nd_item<1> item,
                             dataT* BX, dataT* BY, dataT* BZ,
                             dataT* mx, dataT* my, dataT* mz,
                             dataT* Ms_, dataT Ms_mul,
                             dataT* k1_, dataT k1_mul,
                             dataT* k2_, dataT k2_mul,
                             dataT* k3_, dataT k3_mul,
                             dataT* c1x, dataT c1x_mul,
                             dataT* c1y, dataT c1y_mul,
                             dataT* c1z, dataT c1z_mul,
                             dataT* c2x, dataT c2x_mul,
                             dataT* c2y, dataT c2y_mul,
                             dataT* c2z, dataT c2z_mul,
                             size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {

        dataT invMs = inv_Msat<dataT>(Ms_, Ms_mul, gid);
        dataT    k1 = amul<dataT>(k1_, k1_mul, gid);
        k1         *= invMs;
        dataT    k2 = amul<dataT>(k2_, k2_mul, gid);
        k2         *= invMs;
        dataT    k3 = amul<dataT>(k3_, k3_mul, gid);
        k3         *= invMs;

        sycl::vec<dataT, 3> u1 = normalized<dataT>(vmul<dataT>(c1x, c1y, c1z, c1x_mul, c1y_mul, c1z_mul, gid));
        sycl::vec<dataT, 3> u2 = normalized<dataT>(vmul<dataT>(c2x, c2y, c2z, c2x_mul, c2y_mul, c2z_mul, gid));
        sycl::vec<dataT, 3> u3 = sycl::cross(u1, u2); // 3rd axis perpendicular to u1,u2
        sycl::vec<dataT, 3> m  = make_vec3<dataT>(mx[gid], my[gid], mz[gid]);

        dataT u1m = sycl::dot(u1, m);
        dataT u2m = sycl::dot(u2, m);
        dataT u3m = sycl::dot(u3, m);

        sycl::vec<dataT, 3> B = (dataT)(-2.0)*k1*((pow2(u2m) + pow2(u3m)) * (    (u1m) * u1) +
                                                  (pow2(u1m) + pow2(u3m)) * (    (u2m) * u2) +
                                                  (pow2(u1m) + pow2(u2m)) * (    (u3m) * u3))-
                                 (dataT)(2.0)*k2*((pow2(u2m) * pow2(u3m)) * (    (u1m) * u1) +
                                                  (pow2(u1m) * pow2(u3m)) * (    (u2m) * u2) +
                                                  (pow2(u1m) * pow2(u2m)) * (    (u3m) * u3))-
                                 (dataT)(4.0)*k3*((pow4(u2m) + pow4(u3m)) * (pow3(u1m) * u1) +
                                                  (pow4(u1m) + pow4(u3m)) * (pow3(u2m) * u2) +
                                                  (pow4(u1m) + pow4(u2m)) * (pow3(u3m) * u3));

        // Store to global buffer
        BX[gid] = B.x();
        BY[gid] = B.y();
        BZ[gid] = B.z();
    }
}

// the function that launches the kernel
template <typename dataT>
void addcubicanisotropy2_t(size_t blocks, size_t threads, sycl::queue q,
                           dataT* BX, dataT* BY, dataT* BZ,
                           dataT* mx, dataT* my, dataT* mz,
                           dataT* Ms_, dataT Ms_mul,
                           dataT* k1_, dataT k1_mul,
                           dataT* k2_, dataT k2_mul,
                           dataT* k3_, dataT k3_mul,
                           dataT* c1x_, dataT c1x_mul,
                           dataT* c1y_, dataT c1y_mul,
                           dataT* c1z_, dataT c1z_mul,
                           dataT* c2x_, dataT c2x_mul,
                           dataT* c2y_, dataT c2y_mul,
                           dataT* c2z_, dataT c2z_mul,
                           size_t N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(addcubicanisotropy2_fcn<dataT>, totalThreads, threads,
                             BX, BY, BZ,
                             mx, my, mz,
                             Ms_, Ms_mul,
                             k1_, k1_mul,
                             k2_, k2_mul,
                             k3_, k3_mul,
                             c1x_, c1x_mul,
                             c1y_, c1y_mul,
                             c1z_, c1z_mul,
                             c2x_, c2x_mul,
                             c2y_, c2y_mul,
                             c2z_, c2z_mul,
                             N);
}
