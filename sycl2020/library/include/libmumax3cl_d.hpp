typedef double real_t;

#ifdef __cplusplus
  class Mumax3clUtil {
    public :
      Mumax3clUtil(int id);
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
