#include <CL/sycl.hpp>
#include "libmumax3cl_f.hpp"

int main() {
    auto obj = new Mumax3clUtil(0);
    std::cout << "Working on device: " << obj->getDevice().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "  Maximum number of Compute Units (CUs): " << obj->getDevice().get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "  Maximum work item dimensions: " << obj->getDevice().get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
    std::cout << "  Maximum work group size: " << obj->getDevice().get_info<sycl::info::device::max_work_group_size>() << std::endl;
//    std::cout << "  : " << obj->getDevice().get_info<sycl::info::device::>() << std::endl;
//    std::cout << "  : " << obj->getDevice().get_info<sycl::info::device::>() << std::endl;

    return 0;
}
