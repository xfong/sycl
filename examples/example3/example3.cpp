#include <iostream>
#include <array>
#include <algorithm>

#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

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

int main(int, char**) {
    // Create data array to operate on
    std::array<int, 4> vals = {{ 1, 2, 3, 4 }};

    // Create command queue to execute kernel
    sycl::queue queue(sycl::default_selector{});

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
