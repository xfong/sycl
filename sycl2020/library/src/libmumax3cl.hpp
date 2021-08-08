#include <CL/sycl.hpp>
#include "device/addexchange.hpp"
#include "device/cubicanisotropy2.hpp"
#include "device/div.hpp"
#include "device/dotproduct.hpp"
#include "device/madd2.hpp"
#include "device/madd3.hpp"

template <typename dataT>
class Mumax3clUtil_t {
    public :
        Mumax3clUtil_t(int id) {
            this->mainDev = getGPU(id);
            this->mainQ = sycl::queue(this->mainDev);
        };
        sycl::queue getQueue() { return this->mainQ; }
        sycl::device getDevice() { return this->mainDev; }
        void addcubicanisotropy2(size_t blocks, size_t threads,
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
        void addexchange(size_t blocks[3], size_t threads[3],
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
        void pointwise_div(size_t blocks, size_t threads,
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
        void dotproduct(size_t blocks, size_t threads,
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
        void madd2(size_t blocks, size_t threads,
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
        void madd3(size_t blocks, size_t threads,
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

    private :
        sycl::queue   mainQ;
        sycl::device  mainDev;
};
