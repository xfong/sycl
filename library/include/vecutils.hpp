// This file implements some common utility functions on vector.
// Author: Mykola Dvornik, Arne Vansteenkiste

#ifndef _VECUTILS_HPP_
#define _VECUTILS_HPP_

// converting set of 3 floats into a 3-component vector
template<typename dataT, typename vecT>
inline vecT make_vec3(dataT a, dataT b, dataT c) {
	return (vecT) {a, b, c};
}

// length of the 3-components vector
template<typename dataT, typename vecT>
inline dataT len(vecT a) {
	return length(a);
}

// returns a normalized copy of the 3-components vector
template<typename dataT, typename vecT>
inline vecT normalized(vecT a){
	dataT veclen = (len(a) != 0.0f) ? ( 1.0f / len(a) ) : 0.0f;
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

template<typename dataT, typename vecT>
inline bool is0(vecT m) {
	return ( (m.x == (dataT)(0.0f)) || (m.y == (dataT)(0.0f)) || (m.z == (dataT)(0.0f)));
}

#endif
