#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

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

	std::cout << "Setting up to execute kernel..." << std::endl;

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

	std::cout << "Entering reudction loop..." << std::endl;

    // Reduction loop
    auto len = arr.size();
    while (len != 1) {
        // divison rounding up
        auto n_wgroups = (len + part_size - 1) / part_size;

		std::cout << "Submitting kernel..." << std::endl;

        // Submit kernel to command queue for execution
        queue.submit([&] (sycl::handler& cgh) {
            // Local memory
            sycl::accessor
            <int32_t,
             1,
             sycl::access::mode::read_write,
             sycl::access::target::local>
            local_mem(sycl::range<1>(wgroup_size), cgh);

            // Global memory
            auto global_mem = buf.get_access<sycl::access::mode::read_write>(cgh);

            // Device kernel
            cgh.parallel_for<class reduction_kernel>(
                sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
                [=] (sycl::nd_item<1> item) {

                // Load from global to local memory
                size_t local_id = item.get_local_linear_id();
                size_t global_id = item.get_global_linear_id();

                local_mem[local_id] = 0;

                if ((2*global_id) < len) {
                    local_mem[local_id] = global_mem[2 * global_id] + global_mem[2 * global_id + 1]; // Add neighboring items
                }

                // Synchronize workitems to lcoal memory access
                item.barrier(sycl::access::fence_space::local_space);

                // Reduce to one element
                for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
                    auto idx = 2 * stride * local_id;
                    if (idx < wgroup_size) {
                        local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
                    }

                    // Synchronize workitems to local memory access
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Write workgroup result to global memory
                if (local_id == 0) {
                    global_mem[item.get_group_linear_id()] = local_mem[0];
                }
            });
        });
		std::cout << "Waiting for kernel execution to end..." << std::endl;
        queue.wait_and_throw();
        len = n_wgroups;
    }

	std::cout << "Exiting..." << std::endl;
    // Get result of reduction and print to screen
    auto acc = buf.get_access<sycl::access::mode::read>();
    std::cout << "Sum: " << acc[0] << std::endl;

    return 0;
}
