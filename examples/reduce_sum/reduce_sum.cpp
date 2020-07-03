#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
#include "reducesum.hpp"

int main(int, char**) {
    // Create data array to operate on
    std::array<int32_t, 16> arr;

    std::mt19937 mt_engine(std::random_device{}());
    std::uniform_int_distribution<int32_t> idist(0, 10);

    // Output random data to screen
    std::cout << "Data: ";
    for (auto& el : arr) {
        el = idist(mt_engine);
        std::cout << el << " ";
    }
    std::cout << std::endl;

    // Create device buffer to hold random data
    sycl::buffer<int32_t, 1> buf(arr.data(), sycl::range<1>(arr.size()));

    // Select OpenCL device and create command queue
    sycl::device device = sycl::default_selector{}.select_device();
    sycl::queue queue(device, [] (sycl::exception_list el) {
        for (auto ex : el) {
            std::rethrow_exception(ex);
        }
    });

    // Find out the workgroup size supported by the OpenCL device
    auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();

    // Error check the workgroup size
    if (wgroup_size % 2 != 0) {
        throw "Work-group size has to be even!";
    }
    auto part_size = wgroup_size * 2;

    // Find out whether local memory is available on the OpenCL device
    auto has_local_mem = device.is_host()
          || (device.get_info<sycl::info::device::local_mem_type>()
          != sycl::info::local_mem_type::none);

    // Find out the local memory size supported on the OpenCL device
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

    // Error check the size of local memory
    if (!has_local_mem
        || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
        throw "Device doesn't have enough local memory!";
    }

	auto result = reducesum_async<int>(queue, &buf, 0, arr.size(), wgroup_size, wgroup_size);
    // Get result of reduction and print to screen
    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << "Sum: " << acc[0] << std::endl;

    return 0;
}
