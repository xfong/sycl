#include "libmumax3cl_f.hpp"

#define DATASIZE 1024

int main() {
    auto obj = new Mumax3clUtil(2);
    float *input1 = sycl::malloc_shared<float>(DATASIZE, obj->getQueue());

    std::cout << "  Working on device: " << obj->getDevice().get_info<sycl::info::device::name>() << std::endl;
    free(input1, obj->getQueue());
    return 0;
}
