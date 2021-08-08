typedef float real_t;

class Mumax3clUtil {
    public :
        Mumax3clUtil(int id);
        void addexchange(size_t blocks[3], size_t threads[3],
                   real_t* Bx, real_t* By, real_t* Bz,
                   real_t* mx, real_t* my, real_t* mz,
                   real_t* Ms, real_t Ms_mul,
                   real_t* aLUT2d,
                   uint8_t* regions,
                   real_t wx, real_t wy, real_t wz,
                   size_t Nx, size_t Ny, size_t Nz,
                   uint8_t PBC);
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
    private :
        Mumax3clUtil_t<real_t>* obj;
};

#ifdef __cplusplus
extern "C" {
#endif

  Mumax3clUtil* newMumax3clUtil(int id);

  void addexchange(Mumax3clUtil* obj, size_t blocks[3], size_t threads[3],
             real_t* Bx, real_t* By, real_t* Bz,
             real_t* mx, real_t* my, real_t* mz,
             real_t* Ms, real_t Ms_mul,
             real_t* aLUT2d,
             uint8_t* regions,
             real_t wx, real_t wy, real_t wz,
             size_t Nx, size_t Ny, size_t Nz,
             uint8_t PBC);

  void pointwise_div(Mumax3clUtil* obj, size_t blocks, size_t threads,
             real_t* dst,
             real_t* ax,
             real_t* bx,
             size_t  N);

  void dotproduct(Mumax3clUtil* obj, size_t blocks, size_t threads,
             real_t* dst,
             real_t  prefactor,
             real_t* src1x,
             real_t* src1y,
             real_t* src1z,
             real_t* src2x,
             real_t* src2y,
             real_t* src2z,
             size_t  N);

  void madd2(Mumax3clUtil* obj, size_t blocks, size_t threads,
             real_t* dst,
             real_t* src1,
             real_t  fac1,
             real_t* src2,
             real_t  fac2,
             size_t  N);

  void madd3(Mumax3clUtil* obj, size_t blocks, size_t threads,
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
