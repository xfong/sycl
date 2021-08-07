#include "gpu_select.hpp"
#include "libmumax3cl.hpp"
#include "libmumax3cl_f.hpp"

Mumax3clUtil::Mumax3clUtil(int id) {
    this->obj = new Mumax3clUtil_t<real_t>(id);
}

void Mumax3clUtil::madd2(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t* src1,
                   real_t fac1,
                   real_t* src2,
                   real_t fac2,
                   size_t N) {
    this->obj->madd2(blocks, threads,
                     dst,
                     src1,
                     fac1,
                     src2,
                     fac2,
                     N);
}

void Mumax3clUtil::madd3(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t* src1,
                   real_t fac1,
                   real_t* src2,
                   real_t fac2,
                   real_t* src3,
                   real_t fac3,
                   size_t N) {
    this->obj->madd3(blocks, threads,
                     dst,
                     src1,
                     fac1,
                     src2,
                     fac2,
                     src3,
                     fac3,
                     N);
}

sycl::queue Mumax3clUtil::getQueue() { return this->obj->getQueue(); }
sycl::device Mumax3clUtil::getDevice() { return this->obj->getDevice(); }

// C interface functions
#ifdef __cplusplus
extern "C" {
#endif

Mumax3clUtil* newMumax3clUtil(int id) {
    return new Mumax3clUtil(id);
}

void madd2(Mumax3clUtil* obj, size_t blocks, size_t threads,
      real_t* dst,
      real_t* src1,
      real_t fac1,
      real_t* src2,
      real_t fac2,
      size_t N){
    obj->madd2(blocks, threads,
               dst,
               src1,
               fac1,
               src2,
               fac2,
               N);
}

void madd3(Mumax3clUtil* obj, size_t blocks, size_t threads,
      real_t* dst,
      real_t* src1,
      real_t fac1,
      real_t* src2,
      real_t fac2,
      real_t* src3,
      real_t fac3,
      size_t N){
    obj->madd3(blocks, threads,
               dst,
               src1,
               fac1,
               src2,
               fac2,
               src3,
               fac3,
               N);
}

#ifdef __cplusplus
}
#endif
