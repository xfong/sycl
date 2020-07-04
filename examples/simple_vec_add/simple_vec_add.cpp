#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "gen_device_queue.hpp"

#ifndef IOSTREAM__
#define IOSTREAM__
#include <iostream>
#endif // IOSTREAM__

class vector_addition; // class is defined for kernel

int main(int argc, char** argv) {
	int gpu_num = grabOpts(argc, argv);

    // Data to process
    sycl::float4 a = { 1.0, 2.0, 3.0, 4.0 };
    sycl::float4 b = { 4.0, 3.0, 2.0, 1.0 };
    sycl::float4 c = { 0.0, 0.0, 0.0, 0.0 };

    // Need to select the OpenCL device to use first
    sycl::default_selector device_selector;

    // Then, set up command queue on OpenCL device
    sycl::queue queue = createSYCLqueue(gpu_num);
	std::cout << "Executing on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Start new scope for OpenCL kernel
    {
        // Create memory buffers that the OpenCL will access
        // The template is cl::sycl:buffer<type, dims>
        // The constructor takes a host pointer to the host memory location where
        // associated buffer data is held, and the number of elements
        sycl::buffer<sycl::float4, 1> a_sycl(&a, sycl::range<1>(1));
        sycl::buffer<sycl::float4, 1> b_sycl(&b, sycl::range<1>(1));
        sycl::buffer<sycl::float4, 1> c_sycl(&c, sycl::range<1>(1));

        // Submit kernel to command queue for execution
        queue.submit([&] (sycl::handler& cgh) {
            // Set up accessors for kernel to access memory buffers
            auto a_acc = a_sycl.get_access<sycl::access::mode::read>(cgh); // Data to read in
            auto b_acc = b_sycl.get_access<sycl::access::mode::read>(cgh); // Data to read in
            auto c_acc = c_sycl.get_access<sycl::access::mode::discard_write>(cgh); // Data to overwrite

            // Kernel code
            cgh.single_task<class vector_addition>([=] () {
                c_acc[0] = a_acc[0] + b_acc[0];
            });
        });
    }

    // Will access buffer data so sycl runtime will wait for kernel execution to end and copy
    // data from device buffer back to host memory for output
    std::cout << " A { " << a.x() << ", " << a.y() << ", " << a.z() << ", " << a.w() << " }\n"
              << " B { " << b.x() << ", " << b.y() << ", " << b.z() << ", " << b.w() << " }\n"
              << "-----------------\n"
              << " C { " << c.x() << ", " << c.y() << ", " << c.z() << ", " << c.w() << " }\n"
              << std::endl;

    return 0;
}
