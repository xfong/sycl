typedef float real_t;

#ifdef __cplusplus
  class Mumax3clUtil {
    public :
      Mumax3clUtil(int id);
      void madd2(size_t blocks, size_t threads,
                 real_t* dst,
                 real_t* src1,
                 real_t fac1,
                 real_t* src2,
                 real_t fac2,
                 size_t N);
      void madd3(size_t blocks, size_t threads,
                 real_t* dst,
                 real_t* src1,
                 real_t fac1,
                 real_t* src2,
                 real_t fac2,
                 real_t* src3,
                 real_t fac3,
                 size_t N);
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

  extern void madd2(Mumax3clUtil* obj, size_t blocks, size_t threads,
               real_t* dst,
               real_t* src1,
               real_t fac1,
               real_t* src2,
               real_t fac2,
               size_t N);

  extern void madd3(Mumax3clUtil* obj, size_t blocks, size_t threads,
               real_t* dst,
               real_t* src1,
               real_t fac1,
               real_t* src2,
               real_t fac2,
               real_t* src3,
               real_t fac3,
               size_t N);

#ifdef __cplusplus
}
#endif
