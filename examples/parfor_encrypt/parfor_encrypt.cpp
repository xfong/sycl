#include <iostream>
#include <cstring>
#include <vector>

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

int main(int, char**) {
    // Create text array to operate on
    char text[] = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc interdum in erat non scelerisque.";
    const size_t len = sizeof(text);

    // Create command queue to execute kernel
    sycl::queue queue(sycl::default_selector{});

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
