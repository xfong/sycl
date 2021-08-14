#include "gpu_select.hpp"
#include "libmumax3cl.hpp"
#include "libmumax3cl_d.hpp"

Mumax3clUtil::Mumax3clUtil(int id) {
    this->obj = new Mumax3clUtil_t<real_t>(id);
}

void Mumax3clUtil::copypadmul2(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::copyunpad(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::crop(dim3 blocks, dim3 threads,
          real_t* dst,
          size_t   Dx, size_t   Dy, size_t   Dz,
          real_t* src,
          size_t   Sx, size_t   Sy, size_t   Sz,
          size_t Offx, size_t Offy, size_t Offz) {
    this->obj->crop(blocks, threads,
                    dst,
                    Dx, Dy, Dz,
                    src,
                    Sx, Sy, Sz,
                    Offx, Offy, Offz);
}

void Mumax3clUtil::crossproduct(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::addcubicanisotropy2(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::pointwise_div(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::adddmi(dim3 blocks, dim3 threads,
                   real_t* Hx, real_t* Hy, real_t* Hz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms_, real_t Ms_mul,
                   real_t* aLUT2d, real_t* dLUT2d,
                   uint8_t* regions,
                   size_t cx, size_t cy, size_t cz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC, uint8_t OpenBC){
    this->obj->adddmi(blocks, threads,
                      Hx, Hy, Hz,
                      mx, my, mz,
                      Ms_, Ms_mul,
                      aLUT2d, dLUT2d,
                      regions,
                      cx, cy, cz,
                      Nx, Ny, Nz,
                      PBC, OpenBC);
}

void Mumax3clUtil::adddmibulk(dim3 blocks, dim3 threads,
                   real_t* Hx, real_t* Hy, real_t* Hz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms_, real_t Ms_mul,
                   real_t* aLUT2d, real_t* DLUT2d,
                   uint8_t* regions,
                   size_t cx, size_t cy, size_t cz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC, uint8_t OpenBC){
    this->obj->adddmibulk(blocks, threads,
                      Hx, Hy, Hz,
                      mx, my, mz,
                      Ms_, Ms_mul,
                      aLUT2d, DLUT2d,
                      regions,
                      cx, cy, cz,
                      Nx, Ny, Nz,
                      PBC, OpenBC);
}

void Mumax3clUtil::dotproduct(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::addexchange(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::exchangedecode(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t* aLUT2d,
                   uint8_t* regions,
                   real_t wx, real_t wy, real_t wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC) {
    this->obj->exchangedecode(blocks, threads,
                              dst,
                              aLUT2d,
                              regions,
                              wx, wy, wz,
                              Nx, Ny, Nz,
                              PBC);
}

void Mumax3clUtil::kernmulc(dim3 blocks, dim3 threads,
                   real_t* fftM, real_t* fftK,
                   size_t    Nx, size_t    Ny) {
    this->obj->kernmulc(blocks, threads,
                        fftM, fftK,
                          Nx,   Ny);
            };

void Mumax3clUtil::kernmulrsymm2dxy(dim3 blocks, dim3 threads,
                   real_t*  fftMx, real_t*  fftMy,
                   real_t* fftKxx, real_t* fftKyy, real_t* fftKxy,
                   size_t      Nx, size_t      Ny) {
    this->obj->kernmulrsymm2dxy(blocks, threads,
                                 fftMx,  fftMy,
                                fftKxx, fftKyy, fftKxy,
                                    Nx,     Ny);
            };

void Mumax3clUtil::kernmulrsymm2dz(dim3 blocks, dim3 threads,
                   real_t* fftMz, real_t* fftKzz,
                   size_t     Nx, size_t      Ny) {
    this->obj->kernmulrsymm2dz(blocks, threads,
                               fftMz, fftKzz,
                                  Nx,     Ny);

            };

void Mumax3clUtil::kernmulrsymm3d(dim3 blocks, dim3 threads,
                   real_t*  fftMx, real_t*  fftMy, real_t*  fftMz,
                   real_t* fftKxx, real_t* fftKyy, real_t* fftKzz,
                   real_t* fftKyz, real_t* fftKxz, real_t* fftKxy,
                   size_t      Nx, size_t      Ny, size_t      Nz) {
    this->obj->kernmulrsymm3d(blocks, threads,
                               fftMx,   fftMy,   fftMz,
                              fftKxx,  fftKyy,  fftKzz,
                              fftKyz,  fftKxz,  fftKxy,
                                  Nx,      Ny,      Nz);
            };

void Mumax3clUtil::llnoprecess(dim3 blocks, dim3 threads,
                   real_t*  tx, real_t*  ty, real_t*  tz,
                   real_t* mx_, real_t* my_, real_t* mz_,
                   real_t* hx_, real_t* hy_, real_t* hz_,
                   size_t N) {
    this->obj->llnoprecess(blocks, threads,
                            tx,  ty,  tz,
                           mx_, my_, mz_,
                           hx_, hy_, hz_,
                           N);
}

void Mumax3clUtil::lltorque2(dim3 blocks, dim3 threads,
                  real_t*  tx, real_t*  ty, real_t*  tz,
                  real_t* mx_, real_t* my_, real_t* mz_,
                  real_t* hx_, real_t* hy_, real_t* hz_,
                  real_t* alpha_, real_t alpha_mul,
                  size_t N) {
    this->obj->lltorque2(blocks, threads,
                          tx,  ty,  tz,
                         mx_, my_, mz_,
                         hx_, hy_, hz_,
                         alpha_, alpha_mul,
                         N);
}

void Mumax3clUtil::madd2(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::madd3(dim3 blocks, dim3 threads,
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

void Mumax3clUtil::getmagnetoelasticfield(dim3 blocks, dim3 threads,
                   real_t*  Bx, real_t*     By, real_t* Bz,
                   real_t*  mx, real_t*     my, real_t* mz,
                   real_t* exx, real_t exx_mul,
                   real_t* eyy, real_t eyy_mul,
                   real_t* ezz, real_t ezz_mul,
                   real_t* exy, real_t exy_mul,
                   real_t* exz, real_t exz_mul,
                   real_t* eyz, real_t eyz_mul,
                   real_t*  B1, real_t  B1_mul,
                   real_t*  B2, real_t  B2_mul,
                   real_t*  Ms, real_t  Ms_mul,
                   size_t   N) {
    this->obj->getmagnetoelasticfield(blocks, threads,
                                       Bx,      By, Bz,
                                       mx,      my, mz,
                                      exx, exx_mul,
                                      eyy, eyy_mul,
                                      ezz, ezz_mul,
                                      exy, exy_mul,
                                      exz, exz_mul,
                                      eyz, eyz_mul,
                                       B1,  B1_mul,
                                       B2,  B2_mul,
                                       Ms,  Ms_mul,
                                        N);
}

void Mumax3clUtil::getmagnetoelasticforce(dim3 blocks, dim3 threads,
                   real_t*  fx, real_t*    fy, real_t*  fz,
                   real_t*  mx, real_t*    my, real_t*  mz,
                   real_t* B1_, real_t B1_mul,
                   real_t* B2_, real_t B2_mul,
                   real_t rcsx, real_t   rcsy, real_t rcsz,
                   size_t  Nx, size_t    Ny, size_t  Nz,
                   uint8_t PBC) {
    this->obj->getmagnetoelasticforce(blocks, threads,
                                        fx,     fy,   fz,
                                        mx,     my,   mz,
                                       B1_, B1_mul,
                                       B2_, B2_mul,
                                      rcsx,   rcsy, rcsz,
                                        Nx,     Ny,   Nz,
                                       PBC);
}

void Mumax3clUtil::setmaxangle(dim3 blocks, dim3 threads,
                   real_t*       dst,
                   real_t*        mx, real_t* my, real_t* mz,
                   real_t*    aLUT2d,
                   uint8_t* regions,
                   size_t        Nx, size_t Ny, size_t Nz,
                   uint8_t      PBC) {
    this->obj->setmaxangle(blocks, threads,
                              dst,
                               mx, my, mz,
                           aLUT2d,
                          regions,
                               Nx, Ny, Nz,
                              PBC);
}

void Mumax3clUtil::minimize(dim3 blocks, dim3 threads,
                   real_t*  mx_, real_t*  my_, real_t*  mz_,
                   real_t* m0x_, real_t* m0y_, real_t* m0z_,
                   real_t*  tx_, real_t*  ty_, real_t*  tz_,
                   real_t dt, size_t N) {
    this->obj->minimize(blocks, threads,
                         mx_,  my_,  mz_,
                        m0x_, m0y_, m0z_,
                         tx_,  ty_,  tz_,
                          dt,    N);
}

void Mumax3clUtil::mul(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t*  a0,
                   real_t*  b0,
                   size_t   N) {
    this->obj->mul(blocks, threads,
                   dst,
                    a0,
                    b0,
                     N);
}

void Mumax3clUtil::normalize(dim3 blocks, dim3 threads,
                   real_t* vx, real_t* vy, real_t* vz,
                   real_t* vol,
                   size_t N) {
    this->obj->normalize(blocks, threads,
                          vx, vy, vz,
                         vol,
                          N);
}

void Mumax3clUtil::addoommfslonczewskitorque(dim3 blocks, dim3 threads,
                   real_t*         tx, real_t*             ty, real_t* tz,
                   real_t*         mx, real_t*             my, real_t* mz,
                   real_t*         Ms, real_t          Ms_mul,
                   real_t*         jz, real_t          jz_mul,
                   real_t*         px, real_t          px_mul,
                   real_t*         py, real_t          py_mul,
                   real_t*         pz, real_t          pz_mul,
                   real_t*      alpha, real_t       alpha_mul,
                   real_t*       pfix, real_t        pfix_mul,
                   real_t*      pfree, real_t       pfree_mul,
                   real_t*  lambdafix, real_t   lambdafix_mul,
                   real_t* lambdafree, real_t  lambdafree_mul,
                   real_t*   epsPrime, real_t    epsPrime_mul,
                   real_t*        flt, real_t         flt_mul,
                   size_t          N) {
    this->obj->addoommfslonczewskitorque(blocks, threads,
                                                 tx,             ty, tz,
                                                 mx,             my, mz,
                                                 Ms,         Ms_mul,
                                                 jz,         jz_mul,
                                                 px,         pz_mul,
                                                 py,         py_mul,
                                                 pz,         pz_mul,
                                              alpha,      alpha_mul,
                                               pfix,       pfix_mul,
                                              pfree,      pfree_mul,
                                          lambdafix,  lambdafix_mul,
                                         lambdafree, lambdafree_mul,
                                           epsPrime,   epsPrime_mul,
                                                flt,        flt_mul,
                                                  N);
}

void Mumax3clUtil::regionadds(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*      LUT,
                   uint8_t* regions,
                   size_t         N) {
    this->obj->regionadds(blocks, threads,
                          dst,
                          LUT,
                          regions,
                          N);
}

void Mumax3clUtil::regionaddv(dim3 blocks, dim3 threads,
                   real_t*     dstx, real_t* dsty,  real_t* dstz,
                   real_t*     LUTx, real_t* LUTy,  real_t* LUTz,
                   uint8_t* regions,
                   size_t         N) {
    this->obj->regionaddv(blocks, threads,
                          dstx, dsty, dstz,
                          LUTx, LUTy, LUTz,
                          regions,
                          N);
}

void Mumax3clUtil::regiondecode(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*      LUT,
                   uint8_t* regions,
                   size_t         N) {
    this->obj->regiondecode(blocks, threads,
                            dst,
                            LUT,
                            regions,
                            N);
}

void Mumax3clUtil::regionselect(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*      src,
                   uint8_t* regions,
                   uint8_t   region,
                   size_t         N) {
    this->obj->regionselect(blocks, threads,
                            dst,
                            src,
                            regions,
                            region,
                            N);
}

void Mumax3clUtil::resize(dim3 blocks, dim3 threads,
                   real_t*    dst,
                   size_t      Dx, size_t Dy, size_t Dz,
                   real_t*    src,
                   size_t      Sx, size_t Sy, size_t Sz,
                   int      layer,
                   int     scalex, int scaley) {
    this->obj->resize(blocks, threads,
                      dst,
                      Dx, Dy, Dz,
                      src,
                      Sx, Sy, Sz,
                      layer,
                      scalex, scaley);
}

void Mumax3clUtil::shiftbytesy(dim3 blocks, dim3 threads,
                   uint8_t*    dst,
                   uint8_t*    src,
                   size_t       Nx, size_t Ny, size_t Nz,
                   size_t      shy,
                   uint8_t  clampV) {
    this->obj->shiftbytesy(blocks, threads,
                           dst,
                           src,
                           Nx, Ny, Nz,
                           shy,
                           clampV);
}

void Mumax3clUtil::shiftx(dim3 blocks, dim3 threads,
                   real_t*    dst,
                   real_t*    src,
                   size_t      Nx, size_t    Ny, size_t Nz,
                   size_t     shx,
                   real_t  clampL, real_t clampR) {
    this->obj->shiftx(blocks, threads,
                      dst,
                      src,
                      Nx, Ny, Nz,
                      shx,
                      clampL, clampR);
}

void Mumax3clUtil::shifty(dim3 blocks, dim3 threads,
                   real_t*    dst,
                   real_t*    src,
                   size_t      Nx, size_t    Ny, size_t Nz,
                   size_t     shy,
                   real_t  clampL, real_t clampR) {
    this->obj->shifty(blocks, threads,
                      dst,
                      src,
                      Nx, Ny, Nz,
                      shy,
                      clampL, clampR);
}

void Mumax3clUtil::shiftz(dim3 blocks, dim3 threads,
                   real_t*    dst,
                   real_t*    src,
                   size_t      Nx, size_t    Ny, size_t Nz,
                   size_t     shz,
                   real_t  clampL, real_t clampR) {
    this->obj->shiftz(blocks, threads,
                      dst,
                      src,
                      Nx, Ny, Nz,
                      shz,
                      clampL, clampR);
}

void Mumax3clUtil::addslonczewskitorque2(dim3 blocks, dim3 threads,
                   real_t*       tx, real_t*           ty, real_t* tz,
                   real_t*       mx, real_t*           my, real_t* mz,
                   real_t*       Ms, real_t        Ms_mul,
                   real_t*       jz, real_t        jz_mul,
                   real_t*       px, real_t        px_mul,
                   real_t*       py, real_t        py_mul,
                   real_t*       pz, real_t        pz_mul,
                   real_t*    alpha, real_t     alpha_mul,
                   real_t*      pol, real_t       pol_mul,
                   real_t*   lambda, real_t    lambda_mul,
                   real_t* epsPrime, real_t  epsPrime_mul,
                   real_t*      flt, real_t       flt_mul,
                   size_t         N) {
    this->obj->addslonczewskitorque2(blocks, threads,
                                     tx, ty, tz,
                                     mx, my, mz,
                                     Ms, Ms_mul,
                                     jz, jz_mul,
                                     px, px_mul,
                                     py, py_mul,
                                     pz, pz_mul,
                                     alpha, alpha_mul,
                                     pol, pol_mul,
                                     lambda, lambda_mul,
                                     epsPrime, epsPrime_mul,
                                     flt, flt_mul,
                                     N);
}

void Mumax3clUtil::settemperature2(dim3 blocks, dim3 threads,
                   real_t*            B,
                   real_t*        noise,
                   real_t  kB2_VgammaDt,
                   real_t*           Ms, real_t    Ms_mul,
                   real_t*         temp, real_t  temp_mul,
                   real_t*        alpha, real_t alpha_mul,
                   size_t             N) {
    this->obj->settemperature2(blocks, threads,
                               B,
                               noise,
                               kB2_VgammaDt,
                               Ms, Ms_mul,
                               temp, temp_mul,
                               alpha, alpha_mul,
                               N);
}

void Mumax3clUtil::vecnorm(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t*  a0, real_t* a1, real_t* a2,
                   size_t    N) {
    this->obj->vecnorm(blocks, threads,
                       dst,
                       a0, a1, a2,
                       N);
}

void Mumax3clUtil::zeromask(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*  maskLUT,
                   uint8_t* regions,
                   size_t         N) {
    this->obj->zeromask(blocks, threads,
                        dst,
                        maskLUT,
                        regions,
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

void addexchange(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void addcubicanisotropy2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void copypadmul2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void copyunpad(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void crop(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
          real_t* dst,
          size_t   Dx, size_t   Dy, size_t   Dz,
          real_t* src,
          size_t   Sx, size_t   Sy, size_t   Sz,
          size_t Offx, size_t Offy, size_t Offz) {
    obj->crop(blocks, threads,
              dst,
              Dx, Dy, Dz,
              src,
              Sx, Sy, Sz,
              Offx, Offy, Offz);
}

void crossproduct(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void pointwise_div(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void adddmi(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
                real_t* Hx, real_t* Hy, real_t* Hz,
                real_t* mx, real_t* my, real_t* mz,
                real_t* Ms_, real_t Ms_mul,
                real_t* aLUT2d, real_t* dLUT2d,
                uint8_t* regions,
                size_t cx, size_t cy, size_t cz,
                size_t Nx, size_t Ny, size_t Nz,
                uint8_t PBC, uint8_t OpenBC){
    obj->adddmi(blocks, threads,
                Hx, Hy, Hz,
                mx, my, mz,
                Ms_, Ms_mul,
                aLUT2d, dLUT2d,
                regions,
                cx, cy, cz,
                Nx, Ny, Nz,
                PBC, OpenBC);
}

void adddmibulk(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
                real_t* Hx, real_t* Hy, real_t* Hz,
                real_t* mx, real_t* my, real_t* mz,
                real_t* Ms_, real_t Ms_mul,
                real_t* aLUT2d, real_t* DLUT2d,
                uint8_t* regions,
                size_t cx, size_t cy, size_t cz,
                size_t Nx, size_t Ny, size_t Nz,
                uint8_t PBC, uint8_t OpenBC){
    obj->adddmibulk(blocks, threads,
                Hx, Hy, Hz,
                mx, my, mz,
                Ms_, Ms_mul,
                aLUT2d, DLUT2d,
                regions,
                cx, cy, cz,
                Nx, Ny, Nz,
                PBC, OpenBC);
}

void dotproduct(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void exchangedecode(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
      real_t* dst,
      real_t* aLUT2d,
      uint8_t* regions,
      real_t wx, real_t wy, real_t wz,
      size_t Nx, size_t Ny, size_t Nz,
      uint8_t PBC) {
    obj->exchangedecode(blocks, threads,
                        dst,
                        aLUT2d,
                        regions,
                        wx, wy, wz,
                        Nx, Ny, Nz,
                        PBC);
}

void llnoprecess(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
      real_t*  tx, real_t*  ty, real_t*  tz,
      real_t* mx_, real_t* my_, real_t* mz_,
      real_t* hx_, real_t* hy_, real_t* hz_,
      size_t N) {
    obj->llnoprecess(blocks, threads,
                      tx,  ty,  tz,
                     mx_, my_, mz_,
                     hx_, hy_, hz_,
                     N);
}

void lltorque2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
      real_t*  tx, real_t*  ty, real_t*  tz,
      real_t* mx_, real_t* my_, real_t* mz_,
      real_t* hx_, real_t* hy_, real_t* hz_,
      real_t* alpha_, real_t alpha_mul,
      size_t N) {
    obj->lltorque2(blocks, threads,
                    tx,  ty,  tz,
                   mx_, my_, mz_,
                   hx_, hy_, hz_,
                   alpha_, alpha_mul,
                   N);
}

void madd2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void madd3(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
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

void normalize(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
      real_t* vx, real_t* vy, real_t* vz,
      real_t* vol,
      size_t N) {
    obj->normalize(blocks, threads,
                    vx, vy, vz,
                    vol,
                    N);
}

void vecnorm(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
      real_t* dst,
      real_t* a0, real_t* a1, real_t* a2,
      size_t N) {
    obj->vecnorm(blocks, threads,
                 dst,
                 a0, a1, a2,
                 N);
}

#ifdef __cplusplus
}
#endif
