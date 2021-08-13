#include <CL/sycl.hpp>
#include "libmumax3cl_f.hpp"

#define DATASIZE 1024

void madd2(dim3 blocks, dim3 threads, Mumax3clUtil* kern, real_t* dst, real_t* A, real_t aFac, real_t* B, real_t bFac, size_t N) {
    kern->madd2(blocks, threads, dst, A, aFac, B, bFac, N);
}

int main() {
    auto obj = new Mumax3clUtil(0);
    std::cout << "  Working on device: " << obj->getDevice().get_info<sycl::info::device::name>() << std::endl;

    real_t *dst = sycl::malloc_shared<real_t>(DATASIZE, obj->getQueue());
    real_t *gold = static_cast<real_t *>(malloc(DATASIZE * sizeof(real_t)));
    real_t *input1 = sycl::malloc_shared<real_t>(DATASIZE, obj->getQueue());
    real_t *input2 = sycl::malloc_shared<real_t>(DATASIZE, obj->getQueue());
    real_t fac1 = 0.5;
    real_t fac2 = 0.25;

    for (int i = 0; i < DATASIZE; i++) {
        input1[i] = 1.0 + (real_t)(i);
        input2[i] = 0.1 + (real_t)(i);
        gold[i] = fac1*input1[i] + fac2*input2[i];
    }

    madd2(dim3(1), dim3(128), obj, dst, input1, fac1, input2, fac2, DATASIZE);
    obj->getQueue().wait();

    size_t chk = 0;
    for (int i = 0; i < DATASIZE; i++) {
        if (gold[i] != dst[i]) {
            chk++;
        }
    }
    if (chk == 0) {
        std::cout << "All correct!" << std::endl;
    } else {
        std::cout << "There are " << chk << " errors!" << std::endl;
    }

    free(gold);
    sycl::free(input1, obj->getQueue());
    sycl::free(input2, obj->getQueue());
    sycl::free(dst, obj->getQueue());
    return 0;
}
