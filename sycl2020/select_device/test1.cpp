#include <CL/sycl.hpp>

int main() {
  sycl::queue q;
  for (auto dev : sycl::device::get_devices()) {
    std::cout << "  Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    if (dev.is_gpu()) {
      std::cout << "    This is also a GPU device" << std::endl;
      q = sycl::queue(dev);
    }
  }
  std::cout << "Queue created on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  auto full_device_list = sycl::device::get_devices();
  std::cout << "  Total devices found: " << full_device_list.size() << std::endl;
  std::cout << "  Check device: " << full_device_list[0].get_info<sycl::info::device::name>() << std::endl;
}
