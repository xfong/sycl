// dotproduct kernel

#include "include/device_function.hpp"

// device side function. This is essentially the function of the kernel
// dstX = a1 * b2 - a2 * b1
// dstY = a2 * b0 - a0 * b2
// dstZ = a0 * b1 - a1 * b0
template <typename dataT>
void crossproduct_fcn(size_t totalThreads, sycl::nd_item<1> item,
                      dataT* dstX, dataT* dstY, dataT* dstZ,
                      dataT* a0, dataT* a1, dataT* a2,
                      dataT* b0, dataT* b1, dataT* b2,
                      size_t N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += totalThreads) {
        sycl::vec<dataT, 3> A = sycl::vec<dataT, 3>(a0[gid], a1[gid], a2[gid]);
        sycl::vec<dataT, 3> B = sycl::vec<dataT, 3>(b0[gid], b1[gid], b2[gid]);
        sycl::vec<dataT, 3> AxB = sycl::cross(A, B);
        dstX[gid] = AxB.x();
        dstY[gid] = AxB.y();
        dstZ[gid] = AxB.z();
    }
}

// the function that launches the kernel
template <typename dataT>
void crossproduct_t(size_t blocks, size_t threads, sycl::queue q,
                      dataT* dstX, dataT* dstY, dataT* dstZ,
                      dataT* a0, dataT* a1, dataT* a2,
                      dataT* b0, dataT* b1, dataT* b2,
                      size_t N) {
    libMumax3clDeviceFcnCall(crossproduct_fcn<dataT>, blocks, threads,
                             dstX, dstY, dstZ,
                               a0,   a1,   a2,
                               b0,   b1,   b2,
                             N);
}
