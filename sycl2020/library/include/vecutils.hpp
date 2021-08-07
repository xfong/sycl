// This file implements some common utility functions on vector.
// Author: Mykola Dvornik, Arne Vansteenkiste

#ifndef _VECUTILS_HPP_
#define _VECUTILS_HPP_

// converting set of 3 floats into a 3-component vector
template<typename dataT>
inline sycl::vec<dataT, 3> make_vec3(dataT a, dataT b, dataT c) {
    return sycl::vec<dataT, 3> {a, b, c};
}

// length of the 3-components vector
template<typename dataT>
inline dataT len(sycl::vec<dataT, 3> a) {
    return length(a);
}

// returns a normalized copy of the 3-components vector
template<typename dataT>
inline sycl::vec<dataT, 3> normalized(sycl::vec<dataT, 3> a){
    dataT veclen = (len(a) != (dataT)(0.0)) ? ( (dataT)(1.0) / len(a) ) : (dataT)(0.0);
    return veclen * a;
}

// square
template<typename dataT>
inline dataT pow2(dataT x){
    return x * x;
}


// pow(x, 3)
template<typename dataT>
inline dataT pow3(dataT x){
    return x * x * x;
}


// pow(x, 4)
template<typename dataT>
inline dataT pow4(dataT x){
    dataT s = x*x;
    return s*s;
}

template<typename dataT>
inline bool is0(sycl::vec<dataT, 3> m) {
    return ( (m.x == (dataT)(0.0)) || (m.y == (dataT)(0.0)) || (m.z == (dataT)(0.0)));
}

#endif
