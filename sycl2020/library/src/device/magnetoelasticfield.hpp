// getmagnetoelasticfield kernel
#include "include/amul.hpp"
#include "include/constants.hpp"

// device side function. This is essentially the function of the kernel
// Add magneto-elastic coupling field to B.
// H = - δUmel / δM,
// where Umel is magneto-elastic energy denstiy given by the eq. (12.18) of Gurevich&Melkov "Magnetization Oscillations and Waves", CRC Press, 1996
template <typename dataT>
void getmagnetoelasticfield_fcn(size_t totalThreads, sycl::nd_item<1> item,
                                dataT*  Bx, dataT*     By, dataT* Bz,
                                dataT*  mx, dataT*     my, dataT* mz,
                                dataT* exx, dataT exx_mul,
                                dataT* eyy, dataT eyy_mul,
                                dataT* ezz, dataT ezz_mul,
                                dataT* exy, dataT exy_mul,
                                dataT* exz, dataT exz_mul,
                                dataT* eyz, dataT eyz_mul,
                                dataT*  B1, dataT  B1_mul,
                                dataT*  B2, dataT  B2_mul,
                                dataT*  Ms, dataT  Ms_mul,
                                size_t   N) {
    for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {

        dataT Exx = amul<dataT>(exx_, exx_mul, gid);
        dataT Eyy = amul<dataT>(eyy_, eyy_mul, gid);
        dataT Ezz = amul<dataT>(ezz_, ezz_mul, gid);

        dataT Exy = amul<dataT>(exy_, exy_mul, gid);
        dataT Eyx = Exy;

        dataT Exz = amul<dataT>(exz_, exz_mul, gid);
        dataT Ezx = Exz;

        dataT Eyz = amul<dataT>(eyz_, eyz_mul, gid);
        dataT Ezy = Eyz;

        dataT invMs = inv_Msat<dataT>(Ms_, Ms_mul, gid);

        dataT B1 = amul<dataT>(B1_, B1_mul, gid) * invMs;
        dataT B2 = amul<dataT>(B2_, B2_mul, gid) * invMs;

        sycl::vec<dataT, 3> m = {mx[gid], my[gid], mz[gid]};

        Bx[gid] += -((dataT)(2.0)*B1*m.x()*Exx + B2*(m.y()*Exy + m.z()*Exz));
        By[gid] += -((dataT)(2.0)*B1*m.y()*Eyy + B2*(m.x()*Eyx + m.z()*Eyz));
        Bz[gid] += -((dataT)(2.0)*B1*m.z()*Ezz + B2*(m.x()*Ezx + m.y()*Ezy));
    }
}

// the function that launches the kernel
template <typename dataT>
void getmagnetoelasticfield_t(size_t blocks, size_t threads, sycl::queue q,
                              dataT*  Bx, dataT*     By, dataT* Bz,
                              dataT*  mx, dataT*     my, dataT* mz,
                              dataT* exx, dataT exx_mul,
                              dataT* eyy, dataT eyy_mul,
                              dataT* ezz, dataT ezz_mul,
                              dataT* exy, dataT exy_mul,
                              dataT* exz, dataT exz_mul,
                              dataT* eyz, dataT eyz_mul,
                              dataT*  B1, dataT  B1_mul,
                              dataT*  B2, dataT  B2_mul,
                              dataT*  Ms, dataT  Ms_mul,
                              size_t   N) {
    size_t totalThreads = blocks*threads;
    libMumax3clDeviceFcnCall(getmagnetoelasticfield_fcn<dataT>, totalThreads, threads,
                              Bx,      By,  Bz,
                              mx,      my,  mz,
                             exx, exx_mul,
                             eyy, eyy_mul,
                             ezz, ezz_mul,
                             exy, exy_mul,
                             exz, exz_mul,
                             eyz, eyz_mul,
                              B1,  B1_mul,
                              B2,  B2_mul,
                              Ms,  Ms_mul,
                               N);
}
