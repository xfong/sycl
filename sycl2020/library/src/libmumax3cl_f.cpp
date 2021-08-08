#include "gpu_select.hpp"
#include "libmumax3cl.hpp"
#include "libmumax3cl_f.hpp"

Mumax3clUtil::Mumax3clUtil(int id) {
    this->obj = new Mumax3clUtil_t<real_t>(id);
}

void Mumax3clUtil::addexchange(size_t blocks[3], size_t threads[3],
                   real_t* Bx, real_t* By, real_t* Bz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms, real_t Ms_mul,
                   real_t* aLUT2d,
                   uint8_t* regions,
                   real_t wx, real_t wy, real_t wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC) {
    this->obj->addexchange(blocks, threads,
                           Bx, By, Bz,
                           mx, my, mz,
                           Ms, Ms_mul,
                           aLUT2d, regions,
                           wx, wy, wz,
                           Nx, Ny, Nz,
                           PBC);
}

void Mumax3clUtil::addcubicanisotropy2(size_t blocks, size_t threads,
                   real_t* BX, real_t* BY, real_t* BZ,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms_, real_t Ms_mul,
                   real_t* k1_, real_t k1_mul,
                   real_t* k2_, real_t k2_mul,
                   real_t* k3_, real_t k3_mul,
                   real_t* c1x_, real_t c1x_mul,
                   real_t* c1y_, real_t c1y_mul,
                   real_t* c1z_, real_t c1z_mul,
                   real_t* c2x_, real_t c2x_mul,
                   real_t* c2y_, real_t c2y_mul,
                   real_t* c2z_, real_t c2z_mul,
                   size_t N) {
    this->obj->addcubicanisotropy2(blocks, threads,
                   BX, BY, BZ,
                   mx, my, mz,
                   Ms_, Ms_mul,
                   k1_, k1_mul,
                   k2_, k2_mul,
                   k3_, k3_mul,
                   c1x_, c1x_mul,
                   c1y_, c1y_mul,
                   c1z_, c1z_mul,
                   c2x_, c2x_mul,
                   c2y_, c2y_mul,
                   c2z_, c2z_mul,
                   N);
}

void Mumax3clUtil::copypadmul2(size_t blocks[3], size_t threads[3],
                   real_t* dst,
                   size_t  Dx, size_t Dy, size_t Dz,
                   real_t* src,
                   size_t  Sx, size_t Sy, size_t Sz,
                   real_t* Ms_, size_t Ms_mul,
                   real_t* vol) {
    this->obj->copypadmul2(blocks, threads,
                           dst,
                           Dx, Dy, Dz,
                           src,
                           Sx, Sy, Sz,
                           Ms_, Ms_mul,
                           vol);
}

void Mumax3clUtil::copyunpad(size_t blocks[3], size_t threads[3],
                   real_t* dst,
                   size_t  Dx, size_t Dy, size_t Dz,
                   real_t* src,
                   size_t  Sx, size_t Sy, size_t Sz) {
    this->obj->copyunpad(blocks, threads,
                           dst,
                           Dx, Dy, Dz,
                           src,
                           Sx, Sy, Sz);
}

void Mumax3clUtil::crossproduct(size_t blocks, size_t threads,
                   real_t* dstX, real_t* dstY, real_t* dstZ,
                   real_t*   a0, real_t*   a1, real_t*   a2,
                   real_t*   b0, real_t*   b1, real_t*   b2,
                   size_t  N) {
    this->obj->crossproduct(blocks, threads,
                           dstX, dstY, dstZ,
                             a0,   a1,   a2,
                             b0,   b1,   b2,
                           N);
}

void Mumax3clUtil::pointwise_div(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t* ax,
                   real_t* bx,
                   real_t  N) {
    this->obj->pointwise_div(blocks, threads,
                             dst,
                             ax,
                             bx,
                             N);
}

void Mumax3clUtil::dotproduct(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t  prefactor,
                   real_t* src1x,
                   real_t* src1y,
                   real_t* src1z,
                   real_t* src2x,
                   real_t* src2y,
                   real_t* src2z,
                   size_t  N) {
    this->obj->dotproduct(blocks, threads,
                          dst,
                          prefactor,
                          src1x,
                          src1y,
                          src1z,
                          src2x,
                          src2y,
                          src2z,
                          N);
}

void Mumax3clUtil::madd2(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t* src1,
                   real_t  fac1,
                   real_t* src2,
                   real_t  fac2,
                   size_t  N) {
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
                   real_t  fac1,
                   real_t* src2,
                   real_t  fac2,
                   real_t* src3,
                   real_t  fac3,
                   size_t  N) {
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

void addexchange(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
                 real_t* Bx, real_t* By, real_t* Bz,
                 real_t* mx, real_t* my, real_t* mz,
                 real_t* Ms, real_t Ms_mul,
                 real_t* aLUT2d,
                 uint8_t* regions,
                 real_t wx, real_t wy, real_t wz,
                 size_t Nx, size_t Ny, size_t Nz,
                 uint8_t PBC) {
    obj->addexchange(blocks, threads,
                     Bx, By, Bz,
                     mx, my, mz,
                     Ms, Ms_mul,
                     aLUT2d, regions,
                     wx, wy, wz,
                     Nx, Ny, Nz,
                     PBC);
}

void addcubicanisotropy2(Mumax3clUtil* obj, size_t blocks, size_t threads,
                   real_t* BX, real_t* BY, real_t* BZ,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms_, real_t Ms_mul,
                   real_t* k1_, real_t k1_mul,
                   real_t* k2_, real_t k2_mul,
                   real_t* k3_, real_t k3_mul,
                   real_t* c1x_, real_t c1x_mul,
                   real_t* c1y_, real_t c1y_mul,
                   real_t* c1z_, real_t c1z_mul,
                   real_t* c2x_, real_t c2x_mul,
                   real_t* c2y_, real_t c2y_mul,
                   real_t* c2z_, real_t c2z_mul,
                   size_t N) {
    obj->addcubicanisotropy2(blocks, threads,
                             BX, BY, BZ,
                             mx, my, mz,
                             Ms_, Ms_mul,
                             k1_, k1_mul,
                             k2_, k2_mul,
                             k3_, k3_mul,
                             c1x_, c1x_mul,
                             c1y_, c1y_mul,
                             c1z_, c1z_mul,
                             c2x_, c2x_mul,
                             c2y_, c2y_mul,
                             c2z_, c2z_mul,
                             N);
}

void copypadmul2(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
                 real_t* dst,
                 size_t  Dx, size_t Dy, size_t Dz,
                 real_t* src,
                 size_t  Sx, size_t Sy, size_t Sz,
                 real_t* Ms_, size_t Ms_mul,
                 real_t* vol) {
    obj->copypadmul2(blocks, threads,
                     dst,
                     Dx, Dy, Dz,
                     src,
                     Sx, Sy, Sz,
                     Ms_, Ms_mul,
                     vol);
}

void copyunpad(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
                 real_t* dst,
                 size_t  Dx, size_t Dy, size_t Dz,
                 real_t* src,
                 size_t  Sx, size_t Sy, size_t Sz) {
    obj->copyunpad(blocks, threads,
                     dst,
                     Dx, Dy, Dz,
                     src,
                     Sx, Sy, Sz);
}

void crossproduct(Mumax3clUtil* obj, size_t blocks, size_t threads,
                  real_t* dstX, real_t* dstY, real_t* dstZ,
                  real_t*   a0, real_t*   a1, real_t*   a2,
                  real_t*   b0, real_t*   b1, real_t*   b2,
                  size_t N) {
    obj->crossproduct(blocks, threads,
                      dstX, dstY, dstZ,
                        a0,   a1,   a2,
                        b0,   b1,   b2,
                      N);
}

void pointwise_div(Mumax3clUtil* obj, size_t blocks, size_t threads,
                   real_t* dst,
                   real_t* ax,
                   real_t* bx,
                   size_t  N){
    obj->pointwise_div(blocks, threads,
                       dst,
                       ax,
                       bx,
                       N);
}

void dotproduct(Mumax3clUtil* obj, size_t blocks, size_t threads,
                real_t* dst,
                real_t  prefactor,
                real_t* src1x,
                real_t* src1y,
                real_t* src1z,
                real_t* src2x,
                real_t* src2y,
                real_t* src2z,
                size_t  N){
    obj->dotproduct(blocks, threads,
                    dst,
                    prefactor,
                    src1x,
                    src1y,
                    src1z,
                    src2x,
                    src2y,
                    src2z,
                    N);
}

void madd2(Mumax3clUtil* obj, size_t blocks, size_t threads,
      real_t* dst,
      real_t* src1,
      real_t  fac1,
      real_t* src2,
      real_t  fac2,
      size_t  N){
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
      real_t  fac1,
      real_t* src2,
      real_t  fac2,
      real_t* src3,
      real_t  fac3,
      size_t  N){
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
