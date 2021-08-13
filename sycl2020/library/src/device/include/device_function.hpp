// kernel_launch.hpp
// Macros are defined to ease the launching of kernels, especially
// since most kernel launches use the same format and can be easily
// templated
#ifndef __LIBMUMAX3_KERNEL_LAUNCH__
#define __LIBMUMAX3_KERNEL_LAUNCH__

#define libMumax3clFcnName(...) __VA_ARGS__

#define libMumax3clDeviceFcnCallInt(deviceFnName,numBlocks,numThreads,...) \
    q.parallel_for( \
      syclKernelLaunchGrid(numBlocks,numThreads), \
      [=](sycl::nd_item<3> item) { \
          deviceFnName(item, __VA_ARGS__); \
    });

#define libMumax3clDeviceFcnCall(fcnName,...) libMumax3clDeviceFcnCallInt((fcnName),__VA_ARGS__)

#endif
