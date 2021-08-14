typedef double real_t;

class Mumax3clUtil {
    public :
        Mumax3clUtil(int id);
        void addexchange(dim3 blocks, dim3 threads,
                   real_t* Bx, real_t* By, real_t* Bz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms, real_t Ms_mul,
                   real_t* aLUT2d,
                   uint8_t* regions,
                   real_t wx, real_t wy, real_t wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC);
        void addcubicanisotropy2(dim3 blocks, dim3 threads,
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
                   size_t N);
        void copypadmul2(dim3 blocks, dim3 threads,
                   real_t* dst,
                   size_t  Dx, size_t Dy, size_t Dz,
                   real_t* src,
                   size_t  Sx, size_t Sy, size_t Sz,
                   real_t* Ms_, size_t Ms_mul,
                   real_t* vol);
        void copyunpad(dim3 blocks, dim3 threads,
                   real_t* dst,
                   size_t  Dx, size_t Dy, size_t Dz,
                   real_t* src,
                   size_t  Sx, size_t Sy, size_t Sz);
        void crop(dim3 blocks, dim3 threads,
                   real_t* dst,
                   size_t   Dx, size_t   Dy, size_t   Dz,
                   real_t* src,
                   size_t   Sx, size_t   Sy, size_t   Sz,
                   size_t Offx, size_t Offy, size_t Offz);
        void crossproduct(dim3 blocks, dim3 threads,
                   real_t* dstX, real_t* dstY, real_t* dstZ,
                   real_t*   a0, real_t*   a1, real_t*   a2,
                   real_t*   b0, real_t*   b1, real_t*   b2,
                   size_t N);
        void adddmi(dim3 blocks, dim3 threads,
                   real_t* Hx, real_t* Hy, real_t* Hz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms_, real_t Ms_mul,
                   real_t* aLUT2d, real_t* dLUT2d,
                   uint8_t* regions,
                   size_t cx, size_t cy, size_t cz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC, uint8_t OpenBC);
        void adddmibulk(dim3 blocks, dim3 threads,
                   real_t* Hx, real_t* Hy, real_t* Hz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms_, real_t Ms_mul,
                   real_t* aLUT2d, real_t* DLUT2d,
                   uint8_t* regions,
                   size_t cx, size_t cy, size_t cz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC, uint8_t OpenBC);
        void pointwise_div(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t* ax,
                   real_t* bx,
                   real_t  N);
        void dotproduct(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t  prefactor,
                   real_t* src1x,
                   real_t* src1y,
                   real_t* src1z,
                   real_t* src2x,
                   real_t* src2y,
                   real_t* src2z,
                   size_t  N);
        void exchangedecode(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t* aLUT2d,
                   uint8_t* regions,
                   real_t wx, real_t wy, real_t wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC);
        void kernmulc(dim3 blocks, dim3 threads,
                   real_t* fftM, real_t* fftK,
                   size_t    Nx, size_t    Ny);
        void kernmulrsymm2dxy(dim3 blocks, dim3 threads,
                   real_t*  fftMx, real_t*  fftMy,
                   real_t* fftKxx, real_t* fftKyy, real_t* fftKxy,
                   size_t      Nx, size_t      Ny);
        void kernmulrsymm2dz(dim3 blocks, dim3 threads,
                   real_t* fftMz, real_t* fftKzz,
                   size_t     Nx, size_t      Ny);
        void kernmulrsymm3d(dim3 blocks, dim3 threads,
                   real_t*  fftMx, real_t*  fftMy, real_t*  fftMz,
                   real_t* fftKxx, real_t* fftKyy, real_t* fftKzz,
                   real_t* fftKyz, real_t* fftKxz, real_t* fftKxy,
                   size_t      Nx, size_t      Ny, size_t      Nz);
        void llnoprecess(dim3 blocks, dim3 threads,
                   real_t*  tx, real_t*  ty, real_t*  tz,
                   real_t* mx_, real_t* my_, real_t* mz_,
                   real_t* hx_, real_t* hy_, real_t* hz_,
                   size_t N);
        void lltorque2(dim3 blocks, dim3 threads,
                   real_t*  tx, real_t*  ty, real_t*  tz,
                   real_t* mx_, real_t* my_, real_t* mz_,
                   real_t* hx_, real_t* hy_, real_t* hz_,
                   real_t* alpha_, real_t alpha_mul,
                   size_t N);
        void madd2(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t* src1,
                   real_t  fac1,
                   real_t* src2,
                   real_t  fac2,
                   size_t  N);
        void madd3(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t* src1,
                   real_t  fac1,
                   real_t* src2,
                   real_t  fac2,
                   real_t* src3,
                   real_t  fac3,
                   size_t  N);
        void getmagnetoelasticfield(dim3 blocks, dim3 threads,
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
                   size_t   N);
        void getmagnetoelasticforce(dim3 blocks, dim3 threads,
                              real_t*  fx, real_t*    fy, real_t*  fz,
                              real_t*  mx, real_t*    my, real_t*  mz,
                              real_t* B1_, real_t B1_mul,
                              real_t* B2_, real_t B2_mul,
                              real_t rcsx, real_t   rcsy, real_t rcsz,
                              size_t  Nx, size_t    Ny, size_t  Nz,
                              uint8_t PBC);
        void setmaxangle(dim3 blocks, dim3 threads,
                   real_t*       dst,
                   real_t*        mx, real_t* my, real_t* mz,
                   real_t*    aLUT2d,
                   uint8_t* regions,
                   size_t        Nx, size_t Ny, size_t Nz,
                   uint8_t      PBC);
        void minimize(dim3 blocks, dim3 threads,
                   real_t*  mx_, real_t*  my_, real_t*  mz_,
                   real_t* m0x_, real_t* m0y_, real_t* m0z_,
                   real_t*  tx_, real_t*  ty_, real_t*  tz_,
                   real_t dt, size_t N);
        void mul(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t*  a0,
                   real_t*  b0,
                   size_t   N);
        void normalize(dim3 blocks, dim3 threads,
                   real_t* vx, real_t* vy, real_t* vz,
                   real_t* vol,
                   size_t N);
        void addoommfslonczewskitorque(dim3 blocks, dim3 threads,
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
                   size_t          N);
        void regionadds(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*      LUT,
                   uint8_t* regions,
                   size_t         N);
        void regionaddv(dim3 blocks, dim3 threads,
                   real_t*     dstx, real_t* dsty,  real_t* dstz,
                   real_t*     LUTx, real_t* LUTy,  real_t* LUTz,
                   uint8_t* regions,
                   size_t         N);
        void regiondecode(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*      LUT,
                   uint8_t* regions,
                   size_t         N);
        void regionselect(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*      src,
                   uint8_t* regions,
                   uint8_t   region,
                   size_t         N);
        void resize(dim3 blocks, dim3 threads,
                   real_t*    dst,
                   size_t      Dx, size_t Dy, size_t Dz,
                   real_t*    src,
                   size_t      Sx, size_t Sy, size_t Sz,
                   int      layer,
                   int     scalex, int scaley);
        void vecnorm(dim3 blocks, dim3 threads,
                   real_t* dst,
                   real_t* a0, real_t* a1, real_t* a2,
                   size_t N);
        void zeromask(dim3 blocks, dim3 threads,
                   real_t*      dst,
                   real_t*  maskLUT,
                   uint8_t* regions,
                   size_t         N);
        sycl::queue getQueue();
        sycl::device getDevice();
    private :
        Mumax3clUtil_t<real_t>* obj;
};

#ifdef __cplusplus
extern "C" {
#endif

  Mumax3clUtil* newMumax3clUtil(int id);

  void addexchange(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* Bx, real_t* By, real_t* Bz,
             real_t* mx, real_t* my, real_t* mz,
             real_t* Ms, real_t Ms_mul,
             real_t* aLUT2d,
             uint8_t* regions,
             real_t wx, real_t wy, real_t wz,
             size_t Nx, size_t Ny, size_t Nz,
             uint8_t PBC);

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
             size_t N);

  void copypadmul2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             size_t  Dx, size_t Dy, size_t Dz,
             real_t* src,
             size_t  Sx, size_t Sy, size_t Sz,
             real_t* Ms_, size_t Ms_mul,
             real_t* vol);

  void copyunpad(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             size_t  Dx, size_t Dy, size_t Dz,
             real_t* src,
             size_t  Sx, size_t Sy, size_t Sz);

  void crop(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             size_t   Dx, size_t   Dy, size_t   Dz,
             real_t* src,
             size_t   Sx, size_t   Sy, size_t   Sz,
             size_t Offx, size_t Offy, size_t Offz);

  void crossproduct(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dstX, real_t* dstY, real_t* dstZ,
             real_t*   a0, real_t*   a1, real_t*   a2,
             real_t*   b0, real_t*   b1, real_t*   b2,
             size_t N);

  void pointwise_div(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             real_t* ax,
             real_t* bx,
             size_t  N);

  void adddmi(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* Hx, real_t* Hy, real_t* Hz,
             real_t* mx, real_t* my, real_t* mz,
             real_t* Ms_, real_t Ms_mul,
             real_t* aLUT2d, real_t* dLUT2d,
             uint8_t* regions,
             size_t cx, size_t cy, size_t cz,
             size_t Nx, size_t Ny, size_t Nz,
             uint8_t PBC, uint8_t OpenBC);

  void adddmibulk(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* Hx, real_t* Hy, real_t* Hz,
             real_t* mx, real_t* my, real_t* mz,
             real_t* Ms_, real_t Ms_mul,
             real_t* aLUT2d, real_t* DLUT2d,
             uint8_t* regions,
             size_t cx, size_t cy, size_t cz,
             size_t Nx, size_t Ny, size_t Nz,
             uint8_t PBC, uint8_t OpenBC);

  void dotproduct(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             real_t  prefactor,
             real_t* src1x,
             real_t* src1y,
             real_t* src1z,
             real_t* src2x,
             real_t* src2y,
             real_t* src2z,
             size_t  N);

  void exchangedecode(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             real_t* aLUT2d,
             uint8_t* regions,
             real_t wx, real_t wy, real_t wz,
             size_t Nx, size_t Ny, size_t Nz,
             uint8_t PBC);

  void llnoprecess(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t*  tx, real_t*  ty, real_t*  tz,
             real_t* mx_, real_t* my_, real_t* mz_,
             real_t* hx_, real_t* hy_, real_t* hz_,
             size_t N);

  void lltorque2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t*  tx, real_t*  ty, real_t*  tz,
             real_t* mx_, real_t* my_, real_t* mz_,
             real_t* hx_, real_t* hy_, real_t* hz_,
             real_t* alpha_, real_t alpha_mul,
             size_t N);

  void madd2(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             real_t* src1,
             real_t  fac1,
             real_t* src2,
             real_t  fac2,
             size_t  N);

  void madd3(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             real_t* src1,
             real_t  fac1,
             real_t* src2,
             real_t  fac2,
             real_t* src3,
             real_t  fac3,
             size_t  N);

  void normalize(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* vx, real_t* vy, real_t* vz,
             real_t* vol,
             size_t N);

  void vecnorm(Mumax3clUtil* obj, dim3 blocks, dim3 threads,
             real_t* dst,
             real_t* a0, real_t* a1, real_t* a2,
             size_t N);

#ifdef __cplusplus
}
#endif
