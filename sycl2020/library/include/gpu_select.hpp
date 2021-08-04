#include <CL/sycl.hpp>

sycl::device getGPU(int N) {
  if (N <= 0) {
    return sycl::device(sycl::gpu_selector());
  }

  sycl::device tmp;
  int count = N;
  for (auto dev : sycl::device::get_devices()) {
    if (dev.is_gpu()) {
      --count;
      tmp = dev;
    }
    if (count == 0) {
#ifndef NDEBUG
      std::cout << "Found wanted GPU..." << std::endl;
#endif
      return tmp;
    }
  }
#ifndef NDEBUG
  std::cout << "Number of GPUs are fewer than selection. Returning last found GPU..." << std::endl;
#endif
  return tmp;
}
