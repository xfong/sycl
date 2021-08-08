typedef double real_t;

#ifdef __cplusplus
  class Mumax3clUtil {
    public :
      Mumax3clUtil(int id);
      void addexchange(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
                 real_t* Bx, real_t* By, real_t* Bz,
                 real_t* mx, real_t* my, real_t* mz,
                 real_t* Ms, real_t Ms_mul,
                 real_t* aLUT2d,
                 uint8_t* regions,
                 real_t wx, real_t wy, real_t wz,
                 size_t Nx, size_t Ny, size_t Nz,
                 uint8_t PBC);
      void addcubicanisotropy2(size_t blocks, size_t threads,
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
      void copypadmul2(size_t blocks[3], size_t threads[3],
                 real_t* dst,
                 size_t  Dx, size_t Dy, size_t Dz,
                 real_t* src,
                 size_t  Sx, size_t Sy, size_t Sz,
                 real_t* Ms_, size_t Ms_mul,
                 real_t* vol);
      void pointwise_div(size_t blocks, size_t threads,
                 real_t* dst,
                 real_t* ax,
                 real_t* bx,
                 real_t  N);
      void dotproduct(size_t blocks, size_t threads,
                 real_t* dst,
                 real_t  prefactor,
                 real_t* src1x,
                 real_t* src1y,
                 real_t* src1z,
                 real_t* src2x,
                 real_t* src2y,
                 real_t* src2z,
                 size_t  N);
      void madd2(size_t blocks, size_t threads,
                 real_t* dst,
                 real_t* src1,
                 real_t  fac1,
                 real_t* src2,
                 real_t  fac2,
                 size_t  N);
      void madd3(size_t blocks, size_t threads,
                 real_t* dst,
                 real_t* src1,
                 real_t  fac1,
                 real_t* src2,
                 real_t  fac2,
                 real_t* src3,
                 real_t  fac3,
                 size_t  N);
      sycl::queue getQueue();
      sycl::device getDevice();
  };
#else
  typedef
    struct Mumax3clUtil
      Mumax3clUtil;
#endif

// C interface functions
#ifdef __cplusplus
extern "C" {
#endif

  extern Mumax3clUtil* newMumax3clUtil(int id);

  extern void addexchange(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
               real_t* Bx, real_t* By, real_t* Bz,
               real_t* mx, real_t* my, real_t* mz,
               real_t* Ms, real_t Ms_mul,
               real_t* aLUT2d,
               uint8_t* regions,
               real_t wx, real_t wy, real_t wz,
               size_t Nx, size_t Ny, size_t Nz,
               uint8_t PBC);

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
               size_t N);

  void copypadmul2(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
               real_t* dst,
               size_t  Dx, size_t Dy, size_t Dz,
               real_t* src,
               size_t  Sx, size_t Sy, size_t Sz,
               real_t* Ms_, size_t Ms_mul,
               real_t* vol);

  extern void pointwise_div(Mumax3clUtil* obj, size_t blocks, size_t threads,
               real_t* dst,
               real_t* ax,
               real_t* bx,
               size_t  N);

  extern void dotproduct(Mumax3clUtil* obj, size_t blocks, size_t threads,
               real_t* dst,
               real_t  prefactor,
               real_t* src1x,
               real_t* src1y,
               real_t* src1z,
               real_t* src2x,
               real_t* src2y,
               real_t* src2z,
               size_t  N);

  extern void madd2(Mumax3clUtil* obj, size_t blocks, size_t threads,
               real_t* dst,
               real_t* src1,
               real_t  fac1,
               real_t* src2,
               real_t  fac2,
               size_t  N);

  extern void madd3(Mumax3clUtil* obj, size_t blocks, size_t threads,
               real_t* dst,
               real_t* src1,
               real_t  fac1,
               real_t* src2,
               real_t  fac2,
               real_t* src3,
               real_t  fac3,
               size_t  N);

#ifdef __cplusplus
}
#endif
