#include "gpu_select.hpp"
#include "libmumax3cl.hpp"
#include "libmumax3cl_f.hpp"

Mumax3clUtil::Mumax3clUtil(int id) {
    this->obj = new Mumax3clUtil_t<real_t>(id);
}

void Mumax3clUtil::madd2(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t fac1, real_t* src1,
                   real_t fac2, real_t* src2,
                   size_t N) {
    this->obj->madd2(blocks, threads, dst, fac1, src1, fac2, src2, N);
}

void Mumax3clUtil::madd3(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t fac1, real_t* src1,
                   real_t fac2, real_t* src2,
                   size_t N) {
    this->obj->madd2(blocks, threads, dst, fac1, src1, fac2, src2, N);
}

sycl::queue Mumax3clUtil::getQueue() { return this->obj->getQueue(); }
sycl::device Mumax3clUtil::getDevice() { return this->obj->getDevice(); }
