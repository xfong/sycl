#ifndef __SYCL_UTILS_HPP__
#define __SYCL_UTILS_HPP__

#if defined(__cplusplus)
#include <CL/sycl.hpp>
#endif // __cplusplus

struct dim3 {
    size_t x;
    size_t y;
    size_t z;

#if defined(__cplusplus)
    dim3(size_t _x=1, size_t _y=1, size_t _z=1) : x(_z), y(_y), z(_x) {};
#endif // __cplusplus
};

typedef struct dim3 dim3;

#define     syclBlockIdx_x     item.get_group(2)
#define     syclBlockIdx_y     item.get_group(1)
#define     syclBlockIdx_z     item.get_group(0)

#define     syclBlockDim_x     item.get_local_range(2)
#define     syclBlockDim_y     item.get_local_range(1)
#define     syclBlockDim_z     item.get_local_range(0)

#define    syclThreadIdx_x     item.get_local_id(2)
#define    syclThreadIdx_y     item.get_local_id(1)
#define    syclThreadIdx_z     item.get_local_id(0)

#define    syclThreadCount     item.get_group_range(0)*item.get_local_range(0)*item.get_group_range(1)*item.get_local_range(1)*item.get_group_range(2)*item.get_local_range(2)

#if defined(__cplusplus)
sycl::nd_range<3> syclKernelLaunchGrid(dim3 blocks, dim3 threads) {
    return sycl::nd_range<3>(sycl::range<3>(blocks.x*threads.x, blocks.y*threads.y, blocks.z*threads.z),
                             sycl::range<3>(         threads.x,          threads.y,          threads.z));
}

sycl::nd_range<3> syclKernelLaunchGrid(size_t blocks[3], size_t threads[3]) {
    return sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                             sycl::range<3>(          threads[0],           threads[1],           threads[2]));
}

#endif // __cplusplus

#endif
