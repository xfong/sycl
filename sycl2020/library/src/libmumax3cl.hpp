#include <CL/sycl.hpp>
#include "device/copypadmul2.hpp"
#include "device/copyunpad.hpp"
#include "device/crop.hpp"
#include "device/crossproduct.hpp"
#include "device/cubicanisotropy2.hpp"
#include "device/div.hpp"
#include "device/dmi.hpp"
#include "device/dmibulk.hpp"
#include "device/dotproduct.hpp"
#include "device/exchange.hpp"
#include "device/exchangedecode.hpp"
#include "device/kernmulc.hpp"
#include "device/kernmulrsymm2dxy.hpp"
#include "device/kernmulrsymm2dz.hpp"
#include "device/kernmulrsymm3d.hpp"
#include "device/llnoprecess.hpp"
#include "device/lltorque2.hpp"
#include "device/madd2.hpp"
#include "device/madd3.hpp"
#include "device/magnetoelasticfield.hpp"
#include "device/magnetoelasticforce.hpp"
#include "device/maxangle.hpp"
#include "device/minimize.hpp"
#include "device/mul.hpp"
#include "device/normalize.hpp"
#include "device/oommf_slonczewski.hpp"
#include "device/regionadds.hpp"
#include "device/regionaddv.hpp"
#include "device/regiondecode.hpp"
#include "device/regionselect.hpp"
#include "device/resize.hpp"
#include "device/shiftbytesy.hpp"
#include "device/shiftx.hpp"
#include "device/shifty.hpp"
#include "device/shiftz.hpp"
#include "device/slonczewski2.hpp"
#include "device/temperature2.hpp"
#include "device/topologicalcharge.hpp"
#include "device/uniaxialanisotropy2.hpp"
#include "device/vecnorm.hpp"
#include "device/zeromask.hpp"
#include "device/zhangli2.hpp"
//#include "device/"

template <typename dataT>
class Mumax3clUtil_t {
    public :
        Mumax3clUtil_t(int id) {
            this->mainDev = getGPU(id);
            this->mainQ = sycl::queue(this->mainDev);
        };

        sycl::queue getQueue() { return this->mainQ; }

        sycl::device getDevice() { return this->mainDev; }

        void copypadmul2(dim3 blocks, dim3 threads,
                   dataT* dst,
                   size_t Dx, size_t Dy, size_t Dz,
                   dataT* src,
                   size_t Sx, size_t Sy, size_t Sz,
                   dataT* Ms_, size_t Ms_mul,
                   dataT* vol) {
                copypadmul2_t<dataT>(blocks, threads, this->mainQ,
                   dst,
                   Dx, Dy, Dz,
                   src,
                   Sx, Sy, Sz,
                   Ms_, Ms_mul,
                   vol);
                };

        void copyunpad(dim3 blocks, dim3 threads,
                   dataT* dst,
                   size_t Dx, size_t Dy, size_t Dz,
                   dataT* src,
                   size_t Sx, size_t Sy, size_t Sz) {
                copyunpad_t<dataT>(blocks, threads, this->mainQ,
                   dst,
                   Dx, Dy, Dz,
                   src,
                   Sx, Sy, Sz);
                };

        void crop(dim3 blocks, dim3 threads,
                   dataT*  dst,
                   size_t   Dx, size_t   Dy, size_t   Dz,
                   dataT*  src,
                   size_t   Sx, size_t   Sy, size_t   Sz,
                   size_t Offx, size_t Offy, size_t Offz) {
                crop_t<dataT>(blocks, threads, this->mainQ,
                   dst,
                   Dx, Dy, Dz,
                   src,
                   Sx, Sy, Sz,
                   Offx, Offy, Offz);
                };

        void crossproduct(dim3 blocks, dim3 threads,
                   dataT* dstX, dataT* dstY, dataT* dstZ,
                   dataT*   a0, dataT*   a1, dataT*   a2,
                   dataT*   b0, dataT*   b1, dataT*   b2,
                   size_t N) {
                crossproduct_t<dataT>(blocks, threads, this->mainQ,
                   dstX, dstY, dstZ,
                   a0, a1, a2,
                   b0, b1, b2,
                   N);
                };

        void addcubicanisotropy2(dim3 blocks, dim3 threads,
                   dataT* BX, dataT* BY, dataT* BZ,
                   dataT* mx, dataT* my, dataT* mz,
                   dataT* Ms_, dataT Ms_mul,
                   dataT* k1_, dataT k1_mul,
                   dataT* k2_, dataT k2_mul,
                   dataT* k3_, dataT k3_mul,
                   dataT* c1x_, dataT c1x_mul,
                   dataT* c1y_, dataT c1y_mul,
                   dataT* c1z_, dataT c1z_mul,
                   dataT* c2x_, dataT c2x_mul,
                   dataT* c2y_, dataT c2y_mul,
                   dataT* c2z_, dataT c2z_mul,
                   size_t N) {
                addcubicanisotropy2_t<dataT>(blocks, threads, this->mainQ,
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
                };

        void pointwise_div(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT* ax,
                   dataT* bx,
                   size_t N) {
                pointwise_div_t<dataT>(blocks, threads, this->mainQ,
                                       dst,
                                       ax,
                                       bx,
                                       N);
                };

        void adddmi(dim3 blocks, dim3 threads,
                   dataT* Hx, dataT* Hy, dataT* Hz,
                   dataT* mx, dataT* my, dataT* mz,
                   dataT* Ms_, dataT Ms_mul,
                   dataT* aLUT2d, dataT* dLUT2d,
                   uint8_t* regions,
                   size_t cx, size_t cy, size_t cz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC, uint8_t OpenBC) {
                adddmi_t<dataT>(blocks, threads, this->mainQ,
                                Hx, Hy, Hz,
                                mx, my, mz,
                                Ms_, Ms_mul,
                                aLUT2d, dLUT2d,
                                regions,
                                cx, cy, cz,
                                Nx, Ny, Nz,
                                PBC, OpenBC);
                };

        void adddmibulk(dim3 blocks, dim3 threads,
                   dataT* Hx, dataT* Hy, dataT* Hz,
                   dataT* mx, dataT* my, dataT* mz,
                   dataT* Ms_, dataT Ms_mul,
                   dataT* aLUT2d, dataT* DLUT2d,
                   uint8_t* regions,
                   size_t cx, size_t cy, size_t cz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC, uint8_t OpenBC) {
                adddmibulk_t<dataT>(blocks, threads, this->mainQ,
                                    Hx, Hy, Hz,
                                    mx, my, mz,
                                    Ms_, Ms_mul,
                                    aLUT2d, DLUT2d,
                                    regions,
                                    cx, cy, cz,
                                    Nx, Ny, Nz,
                                    PBC, OpenBC);
                };

        void dotproduct(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT  prefactor,
                   dataT* src1x,
                   dataT* src1y,
                   dataT* src1z,
                   dataT* src2x,
                   dataT* src2y,
                   dataT* src2z,
                   size_t N) {
                dotproduct_t<dataT>(blocks, threads, this->mainQ,
                                    dst,
                                    prefactor,
                                    src1x,
                                    src1y,
                                    src1z,
                                    src2x,
                                    src2y,
                                    src2z,
                                    N);
            };

        void addexchange(dim3 blocks, dim3 threads,
                   dataT* Bx, dataT* By, dataT* Bz,
                   dataT* mx, dataT* my, dataT* mz,
                   dataT* Ms, dataT Ms_mul,
                   dataT* aLUT2d,
                   uint8_t* regions,
                   dataT wx, dataT wy, dataT wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC) {
                addexchange_t<dataT>(blocks, threads, this->mainQ,
                   Bx, By, Bz,
                   mx, my, mz,
                   Ms, Ms_mul,
                   aLUT2d,
                   regions,
                   wx, wy, wz,
                   Nx, Ny, Nz,
                   PBC);
                };

        void exchangedecode(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT* aLUT2d,
                   uint8_t* regions,
                   dataT wx, dataT wy, dataT wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC) {
                exchangedecode_t<dataT>(blocks, threads, this->mainQ,
                                        dst,
                                        aLUT2d,
                                        regions,
                                        wx, wy, wz,
                                        Nx, Ny, Nz,
                                        PBC);
            };

        void kernmulc(dim3 blocks, dim3 threads,
                   dataT* fftM, dataT* fftK,
                   size_t Nx, size_t Ny) {
                kernmulc_t<dataT>(blocks, threads, this->mainQ,
                                  fftM, fftK,
                                    Nx,   Ny);
            };

        void kernmulrsymm2dxy(dim3 blocks, dim3 threads,
                   dataT*  fftMx, dataT*  fftMy,
                   dataT* fftKxx, dataT* fftKyy, dataT* fftKxy,
                   size_t     Nx, size_t     Ny) {
                kernmulrsymm2dxy_t<dataT>(blocks, threads, this->mainQ,
                                           fftMx,  fftMy,
                                          fftKxx, fftKyy, fftKxy,
                                              Nx,     Ny);
            };

        void kernmulrsymm2dz(dim3 blocks, dim3 threads,
                   dataT* fftMz, dataT* fftKzz,
                   size_t    Nx, size_t     Ny) {
                kernmulrsymm2dz_t<dataT>(blocks, threads, this->mainQ,
                                         fftMz, fftKzz,
                                            Nx,     Ny);

            };

        void kernmulrsymm3d(dim3 blocks, dim3 threads,
                   dataT*  fftMx, dataT*  fftMy, dataT*  fftMz,
                   dataT* fftKxx, dataT* fftKyy, dataT* fftKzz,
                   dataT* fftKyz, dataT* fftKxz, dataT* fftKxy,
                   size_t     Nx, size_t     Ny, size_t     Nz) {
                kernmulrsymm3d_t<dataT>(blocks, threads, this->mainQ,
                                         fftMx,   fftMy,   fftMz,
                                        fftKxx,  fftKyy,  fftKzz,
                                        fftKyz,  fftKxz,  fftKxy,
                                            Nx,      Ny,      Nz);
            };

        void llnoprecess(dim3 blocks, dim3 threads,
                   dataT*  tx, dataT*  ty, dataT*  tz,
                   dataT* mx_, dataT* my_, dataT* mz_,
                   dataT* hx_, dataT* hy_, dataT* hz_,
                   size_t N) {
                llnoprecess_t<dataT>(blocks, threads, this->mainQ,
                                      tx,  ty,  tz,
                                     mx_, my_, mz_,
                                     hx_, hy_, hz_,
                                     N);
            };

        void lltorque2(dim3 blocks, dim3 threads,
                   dataT*  tx, dataT*  ty, dataT*  tz,
                   dataT* mx_, dataT* my_, dataT* mz_,
                   dataT* hx_, dataT* hy_, dataT* hz_,
                   dataT* alpha_, dataT alpha_mul,
                   size_t N) {
                lltorque2_t<dataT>(blocks, threads, this->mainQ,
                                      tx,  ty,  tz,
                                     mx_, my_, mz_,
                                     hx_, hy_, hz_,
                                     alpha_, alpha_mul,
                                     N);
            };

        void madd2(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT* src1,
                   dataT fac1,
                   dataT* src2,
                   dataT fac2,
                   size_t N) {
                madd2_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               src1,
                               fac1,
                               src2,
                               fac2,
                               N);
            };

        void madd3(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT* src1,
                   dataT fac1,
                   dataT* src2,
                   dataT fac2,
                   dataT* src3,
                   dataT fac3,
                   size_t N) {
                madd3_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               src1,
                               fac1,
                               src2,
                               fac2,
                               src3,
                               fac3,
                               N);
            };

        void getmagnetoelasticfield(dim3 blocks, dim3 threads,
                   dataT*  Bx, dataT*     By, dataT* Bz,
                   dataT*  mx, dataT*     my, dataT* mz,
                   dataT* exx, dataT exx_mul,
                   dataT* eyy, dataT eyy_mul,
                   dataT* ezz, dataT ezz_mul,
                   dataT* exy, dataT exy_mul,
                   dataT* exz, dataT exz_mul,
                   dataT* eyz, dataT eyz_mul,
                   dataT*  B1, dataT  B1_mul,
                   dataT*  B2, dataT  B2_mul,
                   dataT*  Ms, dataT  Ms_mul,
                   size_t   N) {
                getmagnetoelasticfield_t<dataT>(blocks, threads, this->mainQ,
                               Bx, By, Bz,
                               mx, my, mz,
                               exx, exx_mul,
                               eyy, eyy_mul,
                               ezz, ezz_mul,
                               exy, exy_mul,
                               exz, exz_mul,
                               eyz, eyz_mul,
                               B1, B1_mul,
                               B2, B2_mul,
                               Ms, Ms_mul,
                               N);
            };

        void getmagnetoelasticforce(dim3 blocks, dim3 threads,
                   dataT*  fx, dataT*    fy, dataT*  fz,
                   dataT*  mx, dataT*    my, dataT*  mz,
                   dataT* B1_, dataT B1_mul,
                   dataT* B2_, dataT B2_mul,
                   dataT rcsx, dataT   rcsy, dataT rcsz,
                   size_t  Nx, size_t    Ny, size_t  Nz,
                   uint8_t PBC) {
                getmagnetoelasticforce_t<dataT>(blocks, threads, this->mainQ,
                               fx, fy, fz,
                               mx, my, mz,
                               B1_, B1_mul,
                               B2_, B2_mul,
                               rcsx, rcsy, rcsz,
                               Nx, Ny, Nz,
                               PBC);
            };

        void setmaxangle(dim3 blocks, dim3 threads,
                   dataT*       dst,
                   dataT*        mx, dataT* my, dataT* mz,
                   dataT*    aLUT2d,
                   uint8_t* regions,
                   size_t        Nx, size_t Ny, size_t Nz,
                   uint8_t      PBC) {
                setmaxangle_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               mx, my, mz,
                               aLUT2d,
                               regions,
                               Nx, Ny, Nz,
                               PBC);
            };

        void minimize(dim3 blocks, dim3 threads,
                   dataT*  mx_, dataT*  my_, dataT*  mz_,
                   dataT* m0x_, dataT* m0y_, dataT* m0z_,
                   dataT*  tx_, dataT*  ty_, dataT*  tz_,
                   dataT dt, size_t N) {
                minimize_t<dataT>(blocks, threads, this->mainQ,
                               mx_, my_, mz_,
                               m0x_, m0y_, m0z_,
                               tx_, ty_, tz_,
                               dt, N);
            };

        void mul(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT*  a0,
                   dataT*  b0,
                   size_t   N) {
                mul_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               a0,
                               b0,
                               N);
            };

        void normalize(dim3 blocks, dim3 threads,
                   dataT* vx, dataT* vy, dataT* vz,
                   dataT* vol,
                   size_t N) {
                normalize_t<dataT>(blocks, threads, this->mainQ,
                               vx, vy, vz,
                               vol,
                               N);
            };

        void addoommfslonczewskitorque(dim3 blocks, dim3 threads,
                   dataT*         tx, dataT*             ty, dataT* tz,
                   dataT*         mx, dataT*             my, dataT* mz,
                   dataT*         Ms, dataT          Ms_mul,
                   dataT*         jz, dataT          jz_mul,
                   dataT*         px, dataT          px_mul,
                   dataT*         py, dataT          py_mul,
                   dataT*         pz, dataT          pz_mul,
                   dataT*      alpha, dataT       alpha_mul,
                   dataT*       pfix, dataT        pfix_mul,
                   dataT*      pfree, dataT       pfree_mul,
                   dataT*  lambdafix, dataT   lambdafix_mul,
                   dataT* lambdafree, dataT  lambdafree_mul,
                   dataT*   epsPrime, dataT    epsPrime_mul,
                   dataT*        flt, dataT         flt_mul,
                   size_t          N) {
                addoommfslonczewskitorque_t<dataT>(blocks, threads, this->mainQ,
                               tx, ty, tz,
                               mx, my, mz,
                               Ms, Ms_mul,
                               jz, jz_mul,
                               px, px_mul,
                               py, py_mul,
                               pz, pz_mul,
                               alpha, alpha_mul,
                               pfix, pfix_mul,
                               pfree, pfree_mul,
                               lambdafix, lambdafix_mul,
                               lambdafree, lambdafree_mul,
                               epsPrime, epsPrime_mul,
                               flt, flt_mul,
                               N);
            };

        void regionadds(dim3 blocks, dim3 threads,
                   dataT*       dst,
                   dataT*       LUT,
                   uint8_t* regions,
                   size_t         N) {
                regionadds_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               LUT,
                               regions,
                               N);
            };

        void regionaddv(dim3 blocks, dim3 threads,
                   dataT*       dstx, dataT* dsty,  dataT* dstz,
                   dataT*       LUTx, dataT* LUTy,  dataT* LUTz,
                   uint8_t* regions,
                   size_t         N) {
                regionaddv_t<dataT>(blocks, threads, this->mainQ,
                               dstx, dsty, dstz,
                               LUTx, LUTy, LUTz,
                               regions,
                               N);
            };

        void regiondecode(dim3 blocks, dim3 threads,
                   dataT*       dst,
                   dataT*       LUT,
                   uint8_t* regions,
                   size_t         N) {
                regiondecode_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               LUT,
                               regions,
                               N);
            };

        void regionselect(dim3 blocks, dim3 threads,
                   dataT*       dst,
                   dataT*       src,
                   uint8_t* regions,
                   uint8_t   region,
                   size_t         N) {
                regionselect_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               src,
                               regions,
                               region,
                               N);
            };

        void resize(dim3 blocks, dim3 threads,
                   dataT*    dst,
                   size_t     Dx, size_t Dy, size_t Dz,
                   dataT*    src,
                   size_t     Sx, size_t Sy, size_t Sz,
                   int     layer,
                   int    scalex, int scaley) {
                resize_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               Dx, Dy, Dz,
                               src,
                               Sx, Sy, Sz,
                               layer,
                               scalex, scaley);
            };

        void shiftbytesy(dim3 blocks, dim3 threads,
                   uint8_t*    dst,
                   uint8_t*    src,
                   size_t       Nx, size_t Ny, size_t Nz,
                   size_t      shy,
                   uint8_t  clampV) {
                shiftbytesy_t(blocks, threads, this->mainQ,
                               dst,
                               src,
                               Nx, Ny, Nz,
                               shy,
                               clampV);
            };

        void shiftx(dim3 blocks, dim3 threads,
                   dataT*    dst,
                   dataT*    src,
                   size_t     Nx, size_t    Ny, size_t Nz,
                   size_t    shx,
                   dataT  clampL, dataT clampR) {
                shiftx_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               src,
                               Nx, Ny, Nz,
                               shx,
                               clampL, clampR);
            };

        void shifty(dim3 blocks, dim3 threads,
                   dataT*    dst,
                   dataT*    src,
                   size_t     Nx, size_t    Ny, size_t Nz,
                   size_t    shy,
                   dataT  clampL, dataT clampR) {
                shifty_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               src,
                               Nx, Ny, Nz,
                               shy,
                               clampL, clampR);
            };

        void shiftz(dim3 blocks, dim3 threads,
                   dataT*    dst,
                   dataT*    src,
                   size_t     Nx, size_t    Ny, size_t Nz,
                   size_t    shz,
                   dataT  clampL, dataT clampR) {
                shiftx_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               src,
                               Nx, Ny, Nz,
                               shz,
                               clampL, clampR);
            };

        void addslonczewskitorque2(dim3 blocks, dim3 threads,
                             dataT*       tx, dataT*           ty, dataT* tz,
                             dataT*       mx, dataT*           my, dataT* mz,
                             dataT*       Ms, dataT        Ms_mul,
                             dataT*       jz, dataT        jz_mul,
                             dataT*       px, dataT        px_mul,
                             dataT*       py, dataT        py_mul,
                             dataT*       pz, dataT        pz_mul,
                             dataT*    alpha, dataT     alpha_mul,
                             dataT*      pol, dataT       pol_mul,
                             dataT*   lambda, dataT    lambda_mul,
                             dataT* epsPrime, dataT  epsPrime_mul,
                             dataT*      flt, dataT       flt_mul,
                             size_t        N) {
                addslonczewskitorque2_t<dataT>(blocks, threads, this->mainQ,
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
            };

        void settemperature2(dim3 blocks, dim3 threads,
                       dataT*            B,
                       dataT*        noise,
                       dataT  kB2_VgammaDt,
                       dataT*           Ms, dataT    Ms_mul,
                       dataT*         temp, dataT  temp_mul,
                       dataT*        alpha, dataT alpha_mul,
                       size_t            N) {
                settemperature2_t<dataT>(blocks, threads, this->mainQ,
                               B,
                               noise,
                               kB2_VgammaDt,
                               Ms, Ms_mul,
                               temp, temp_mul,
                               alpha, alpha_mul,
                               N);
            };

        void settopologicalcharge(dim3 blocks, dim3 threads,
                            dataT*    s,
                            dataT*   mx, dataT* my, dataT* mz,
                            dataT icxcy,
                            size_t   Nx, size_t Ny, size_t Nz,
                            uint8_t PBC) {
                settopologicalcharge_t<dataT>(blocks, threads, this->mainQ,
                               s,
                               mx, my, mz,
                               icxcy,
                               Nx, Ny, Nz,
                               PBC);
            };

        void adduniaxialanisotropy2(dim3 blocks, dim3 threads,
                              dataT*  BX, dataT*     BY, dataT* BZ,
                              dataT*  mx, dataT*     my, dataT* mz,
                              dataT* Ms_, dataT  Ms_mul,
                              dataT* k1_, dataT  k1_mul,
                              dataT* k2_, dataT  k2_mul,
                              dataT* ux_, dataT  ux_mul,
                              dataT* uy_, dataT  uy_mul,
                              dataT* uz_, dataT  uz_mul,
                              size_t   N) {
                adduniaxialanisotropy2_t<dataT>(blocks, threads, this->mainQ,
                               BX, BY, BZ,
                               mx, my, mz,
                               Ms_, Ms_mul,
                               k1_, k1_mul,
                               k2_, k2_mul,
                               ux_, ux_mul,
                               uy_, uy_mul,
                               uz_, uz_mul,
                               N);
            };

        void vecnorm(dim3 blocks, dim3 threads,
                   dataT* dst,
                   dataT* a0, dataT* a1, dataT* a2,
                   size_t N) {
                vecnorm_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               a0, a1, a2,
                               N);
            };

        void zeromask(dim3 blocks, dim3 threads,
                   dataT*       dst,
                   dataT*   maskLUT,
                   uint8_t* regions,
                   size_t         N) {
                zeromask_t<dataT>(blocks, threads, this->mainQ,
                               dst,
                               maskLUT,
                               regions,
                               N);
            };

        void addzhanglitorque2(dim3 blocks, dim3 threads,
                   dataT*     TX, dataT*       TY, dataT* TZ,
                   dataT*     mx, dataT*       my, dataT* mz,
                   dataT*    Ms_, dataT    Ms_mul,
                   dataT*    jx_, dataT    jx_mul,
                   dataT*    jy_, dataT    jy_mul,
                   dataT*    jz_, dataT    jz_mul,
                   dataT* alpha_, dataT alpha_mul,
                   dataT*    xi_, dataT    xi_mul,
                   dataT*   pol_, dataT   pol_mul,
                   dataT      cx, dataT        cy, dataT  cz,
                   size_t     Nx, size_t       Ny, size_t Nz,
                   uint8_t   PBC) {
                addzhanglitorque2_t<dataT>(blocks, threads, this->mainQ,
                               TX, TY, TZ,
                               mx, my, mz,
                               Ms_, Ms_mul,
                               jx_, jz_mul,
                               jy_, jy_mul,
                               jz_, jz_mul,
                               alpha_, alpha_mul,
                               xi_, xi_mul,
                               pol_, pol_mul,
                               cx, cy, cz,
                               Nx, Ny, Nz,
                               PBC);
            };

    private :
        sycl::queue   mainQ;
        sycl::device  mainDev;
};
