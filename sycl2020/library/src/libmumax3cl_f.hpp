typedef float real_t;

class Mumax3clUtil {
    public :
        Mumax3clUtil(int id);
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
