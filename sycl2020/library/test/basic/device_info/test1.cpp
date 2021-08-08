#include <CL/sycl.hpp>
#include "libmumax3cl_f.hpp"

int main() {
    auto obj = new Mumax3clUtil(0);
    std::cout << "Working on device: " << obj->getDevice().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "  Maximum number of Compute Units (CUs): " << obj->getDevice().get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "  Maximum work item dimensions: " << obj->getDevice().get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
    auto result = obj->getDevice().get_info<sycl::info::device::max_work_item_sizes>();
    std::cout << "  Maximum work item sizes: [" << result[0] << ", " << result[1] << ", " << result[2] << "]" << std::endl;
    std::cout << "  Maximum work group size: " << obj->getDevice().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "  Preferred vector width (float): " << obj->getDevice().get_info<sycl::info::device::preferred_vector_width_float>() << std::endl;
    std::cout << "  Preferred vector width (double): " << obj->getDevice().get_info<sycl::info::device::preferred_vector_width_double>() << std::endl;
    std::cout << "  Native vector width (float): " << obj->getDevice().get_info<sycl::info::device::native_vector_width_float>() << std::endl;
    std::cout << "  Native vector width (double): " << obj->getDevice().get_info<sycl::info::device::native_vector_width_double>() << std::endl;
    std::cout << "  Maximum memory allocation size (bytes): " << obj->getDevice().get_info<sycl::info::device::max_mem_alloc_size>() << std::endl;
    std::cout << "  Global memory size (bytes): " << obj->getDevice().get_info<sycl::info::device::global_mem_size>() << std::endl;
    std::cout << "  Local memory size (bytes): " << obj->getDevice().get_info<sycl::info::device::local_mem_size>() << std::endl;
    std::cout << "  Global cache memory size (bytes): " << obj->getDevice().get_info<sycl::info::device::global_mem_cache_size>() << std::endl;
    std::cout << "  Global cache line size (bytes): " << obj->getDevice().get_info<sycl::info::device::global_mem_cache_line_size>() << std::endl;
//    std::cout << "  : " << obj->getDevice().get_info<sycl::info::device::>() << std::endl;
//    std::cout << "  : " << obj->getDevice().get_info<sycl::info::device::>() << std::endl;
//    std::cout << "  : " << obj->getDevice().get_info<sycl::info::device::>() << std::endl;
//    std::cout << "  : " << obj->getDevice().get_info<sycl::info::device::>() << std::endl;

    return 0;
}
