#include <CL/sycl.hpp>
#include "gpu_select.hpp"
#include "madd2.hpp"

#define DATASIZE 1024
typedef float real_t;

int main() {
    auto mainQ = sycl::queue(sycl::host_selector());
    auto mainDev = mainQ.get_device();

    std::cout << "  Working on device: " << mainDev.get_info<sycl::info::device::name>() << std::endl;

    real_t *dst = sycl::malloc_shared<real_t>(DATASIZE, mainQ);
    real_t *gold = static_cast<real_t *>(malloc(DATASIZE * sizeof(real_t)));
    real_t *input1 = sycl::malloc_shared<real_t>(DATASIZE, mainQ);
    real_t *input2 = sycl::malloc_shared<real_t>(DATASIZE, mainQ);
    real_t fac1 = 0.5;
    real_t fac2 = 0.25;

    for (int i = 0; i < DATASIZE; i++) {
        input1[i] = 1.0 + (real_t)(i);
        input2[i] = 0.1 + (real_t)(i);
        gold[i] = fac1*input1[i] + fac2*input2[i];
    }

    madd2_t<float>(1, 128, mainQ, dst, fac1, input1, fac2, input2, DATASIZE);
    mainQ.wait();

    size_t chk = 0;
    for (int i = 0; i < DATASIZE; i++) {
        if (gold[i] != dst[i]) {
            chk++;
            std::cout << "    golden[:" << i << "]: " << gold[i] << "; dst[" << i << "]: " << dst[i] << std::endl;
        }
    }
    if (chk == 0) {
        std::cout << "All correct!" << std::endl;
    } else {
        std::cout << "There are " << chk << " errors!" << std::endl;
    }

    free(gold);
    sycl::free(input1, mainQ);
    sycl::free(input2, mainQ);
    sycl::free(dst, mainQ);
    return 0;
}
