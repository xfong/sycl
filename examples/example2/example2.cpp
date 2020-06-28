#include <CL/sycl.hpp>
#include <iostream>

namespace sycl = cl::sycl;

int main(int, char**) {
    // Create the exception handler for demo
    auto exception_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception:\n"
                          << e.what() << std::endl;
            }
        }
    };

    // Create command queue to execute device code
    sycl::queue queue(sycl::default_selector{}, exception_handler);

    // Execute kernel with wrong arguments
    queue.submit([&] (sycl::handler& cgh) {
        auto range = sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(10));
        cgh.parallel_for<class invalid_kernel>(range, [=] (sycl::nd_item<1>) {});
    });

    try {
        queue.wait_and_throw();
    } catch(sycl::exception const& e) {
        std::cout << "Caught synchronous SYCL exception:\n"
                  << e.what() << std::endl;
    }

    return 0;
}
