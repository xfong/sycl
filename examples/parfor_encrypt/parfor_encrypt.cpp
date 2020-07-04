#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "gen_device_queue.hpp"

#ifndef IOSTREAM__
#define IOSTREAM__
#include <iostream>
#endif // IOSTREAM__

#ifndef CSTRING__
#define CSTRING__
#include <cstring>
#include <string>
#endif // CSTRING__

#ifndef VECTOR__
#define VECTOR__
#include <vector>
#endif //VECTOR__

int main(int argc, char** argv) {
	int gpu_num = grabOpts(argc, argv);

    // Create text array to operate on
    char text[] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc interdum in erat non scelerisque.";
    const size_t len = sizeof(text);

    // Create command queue to execute kernel
    sycl::queue queue = createSYCLqueue(gpu_num);
	std::cout << "Executing on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Scope for kernel execution
    {
        // Memory buffer for device side
        sycl::buffer<char, 1> buf(text, sycl::range<1>(len));

        // Submit to command queue for execution
        queue.submit([&] (sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class parrot13>(
                sycl::range<1>(len - 1),
                [=] (sycl::item<1> item) {
                    size_t id = item.get_linear_id();
                    auto const c = acc[id];
                    acc[id] = (c-1/(~(~c|32)/13*2-11)*13);
            });
        });
    }

    // Copy processed text from device to host
    std::cout << text << std::endl;

    return 0;
}
