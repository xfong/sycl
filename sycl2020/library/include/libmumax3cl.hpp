#include <CL/sycl.hpp>
#include "madd2.hpp"
#include "madd3.hpp"

template <typename dataT>
class Mumax3clUtil_t {
    public :
        Mumax3clUtil_t(int id) {
            this->mainDev = getGPU(id);
            this->mainQ = sycl::queue(this->mainDev);
        };
        sycl::queue getQueue() { return this->mainQ; }
        sycl::device getDevice() { return this->mainDev; }
        void madd2(size_t blocks, size_t threads,
                   dataT* dst,
                   dataT fac1, dataT* src1,
                   dataT fac2, dataT* src2,
                   size_t N) {
                madd2_t<dataT>(blocks, threads, this->mainQ, dst, fac1, src1, fac2, src2, N);
            };
        void madd3(size_t blocks, size_t threads,
                   dataT* dst,
                   dataT fac1, dataT* src1,
                   dataT fac2, dataT* src2,
                   dataT fac3, dataT* src3,
                   size_t N) {
                madd3_t<dataT>(blocks, threads, this->mainQ, dst, fac1, src1, fac2, src2, fac3, src3, N);
            };

    private :
        sycl::queue   mainQ;
        sycl::device  mainDev;
};
