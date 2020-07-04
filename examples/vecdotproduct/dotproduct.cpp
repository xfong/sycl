#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>

#include "../../library/include/sycl_engine.hpp"

namespace sycl = cl::sycl;

int main(int, char**) {

	// Initialize random number generator
	srand (static_cast <unsigned> (time(0)));
	
	// Initialize input vectors and output garbage
	const size_t array_size = 2048;
	std::vector<float> A(array_size), B(array_size), C(array_size), D(array_size), E(array_size), F(array_size), G(array_size);
	
	for (size_t idx = 0; idx < array_size; idx++) {
		A[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		B[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		C[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		D[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		E[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		F[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		G[idx] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	
	// Need to select the OpenCL device to use first
	sycl::default_selector device_selector;

	// Then, set up command queue on OpenCL device
	sycl::queue queue(device_selector);

	// Create memory buffers that the OpenCL will access
	// The template is cl::sycl:buffer<type, dims>
	// The constructor takes a host pointer to the host memory location where
	// associated buffer data is held, and the number of elements
	{
		sycl::buffer<sycl::cl_float, 1> a_sycl(A.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> b_sycl(B.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> c_sycl(C.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> d_sycl(D.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> e_sycl(E.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> f_sycl(F.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> g_sycl(G.data(), sycl::range<1>(array_size));

		dotproduct_async<float>(queue, &g_sycl, 1.f, &a_sycl, &b_sycl, &c_sycl, &d_sycl, &e_sycl, &f_sycl, array_size, 1024, 256);
	}
	for (unsigned int i = 0; i < array_size; i++) {
		if ((G[i]*0.000001f) <= fabs(G[i] - (A[i] * D[i]) - (B[i] * E[i]) - (C[i] * F[i]))) {
			std::cout << "The results are incorrect element " << i << " is " << G[i] << "\n"
		    << " but inputs are vector 1: { " << A[i] << ", " << B[i] << ", " << C[i] << " }!\n"
			<< "                vector 2: { " << D[i] << ", " << E[i] << ", " << F[i] << " }!\n"
			<< " expecting result: " << (A[i] * D[i]) + (B[i] * E[i]) + (C[i] * F[i]) << "\n"
			<< " error is " << fabs(G[i] - (A[i] * D[i]) - (B[i] * E[i]) - (C[i] * F[i])) << "\n";
			return 1;
		}
	}
	std::cout << "The results are correct for float dotproduct!\n";

	return 0;
}
