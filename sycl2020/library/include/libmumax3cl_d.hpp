typedef double real_t;

class Mumax3clUtil {
    public :
        Mumax3clUtil(int id);
        void madd2(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t fac1, real_t* src1,
                   real_t fac2, real_t* src2,
                   size_t N);
        void madd3(size_t blocks, size_t threads,
                   real_t* dst,
                   real_t fac1, real_t* src1,
                   real_t fac2, real_t* src2,
                   size_t N);
        sycl::queue getQueue();
        sycl::device getDevice();
};
