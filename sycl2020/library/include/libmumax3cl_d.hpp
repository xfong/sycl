#include "libmumax3cl.hpp"

typedef double real_t;

class Mumax3clUtil {
    public :
        Mumax3clUtil(int id) {
            this->obj = new Mumax3clUtil_t<real_t>(id);
        }
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
    private :
        Mumax3clUtil_t<real_t>* obj;
};
