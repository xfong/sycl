#ifndef _AMUL_H_
#define _AMUL_H_

#include "vecutils.hpp"

// Returns mul * arr[i], or mul when arr == NULL;
template<typename dataT>
inline dataT amul(dataT *arr, dataT mul, int i) {
    return (arr == NULL) ? (mul) : (mul * arr[i]);
}

// Returns m * a[i], or m when a == NULL;
template<typename dataT>
inline sycl::vec<dataT, 3> vmul(dataT &ax,
                                dataT &ay,
                                dataT &az,
                                dataT  mx,
                                dataT  my,
                                dataT  mz,
                                int    i) {
    return make_vec3<dataT>(amul(ax, mx, i),
                            amul(ay, my, i),
                            amul(az, mz, i));
}

// Returns 1/Msat, or 0 when Msat == 0.
template<typename dataT>
inline dataT inv_Msat(dataT& Ms_, dataT Ms_mul, int i) {
    dataT ms = amul(Ms_, Ms_mul, i);
    if (ms == (dataT)(0.0)) {
        return (dataT)(0.0);
    } else {
        return (dataT)(1.0) / ms;
    }
}
#endif
