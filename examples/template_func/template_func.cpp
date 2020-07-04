#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "gen_device_queue.hpp"

#ifndef IOSTREAM__
#define IOSTREAM__
#include <iostream>
#endif // IOSTREAM__

#ifndef ARRAY__
#define ARRAY__
#include <array>
#endif // ARRAY__

#ifndef ALGORITHM__
#define ALGORITHM__
#include <algorithm>
#endif ALGORITHM__

template<typename T, typename Acc, size_t N>
class ConstantAdder {
public:
    ConstantAdder(Acc accessor, T val)
        : accessor(accessor)
        , val(val) {}

    void operator() () {
        for (size_t i = 0; i < N; i++) {
            accessor[i] += val;
        }
    }

private:
    Acc accessor;
    const T val;
};

int main(int argc, char** argv) {
	int gpu_num = grabOpts(argc, argv);

    // Create data array to operate on
    std::array<int, 4> vals = {{ 1, 2, 3, 4 }};

    // Create command queue to execute kernel
    sycl::queue queue = createSYCLqueue(gpu_num);
	std::cout << "Executing on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Scope for kernel execution
    {
        // Memory buffer for device side
        sycl::buffer<int, 1> buf(vals.data(), sycl::range<1>(4));

        // Submit to command queue for execution
        queue.submit([&] (sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);

            cgh.single_task(ConstantAdder<int, decltype(acc), 4>(acc, 1));
        });
    }

    // Copy data from device to host
    std::for_each(vals.begin(), vals.end(), [] (int i) { std::cout << i << " "; } );
    std::cout << std::endl;

    return 0;
}
