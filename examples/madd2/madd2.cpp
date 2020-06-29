#include <vector>
#include <iostream>

#include <CL/sycl.hpp>
#include "madd2.hpp"

namespace sycl = cl::sycl;

int main(int, char**) {
	const size_t array_size = 1024;
	std::vector<int> A(array_size), B(array_size), C(array_size);
	
	std::fill(A.begin(), A.end(), 1);
	std::fill(B.begin(), B.end(), 3);
	std::fill(C.begin(), C.end(), 12);

	std::vector<float> D(array_size), E(array_size), F(array_size);
	
	std::fill(D.begin(), D.end(), 0.5f);
	std::fill(E.begin(), E.end(), 0.25f);
	std::fill(F.begin(), F.end(), 125.5f);

	// Need to select the OpenCL device to use first
	sycl::default_selector device_selector;

	// Then, set up command queue on OpenCL device
	sycl::queue queue(device_selector);

	// Create memory buffers that the OpenCL will access
	// The template is cl::sycl:buffer<type, dims>
	// The constructor takes a host pointer to the host memory location where
	// associated buffer data is held, and the number of elements
	{
		sycl::buffer<sycl::cl_int, 1> a_sycl(A.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_int, 1> b_sycl(B.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_int, 1> c_sycl(C.data(), sycl::range<1>(array_size));

		madd2_async<int>(queue, &c_sycl, &a_sycl, 2, &b_sycl, 3, array_size, array_size, array_size);
	}
	for (unsigned int i = 0; i < array_size; i++) {
		if (C[i] != 2 * A[i] + 3 * B[i]) {
			std::cout << "The results are incorrect element " << i << " is " << C[i]
					  << " but inputs are " << A[i] << " and " << B[i] << "!\n";
			return 1;
		}
	}
	std::cout << "The results are correct for int!\n";

	// Create memory buffers that the OpenCL will access
	// The template is cl::sycl:buffer<type, dims>
	// The constructor takes a host pointer to the host memory location where
	// associated buffer data is held, and the number of elements
	{
		sycl::buffer<sycl::cl_float, 1> d_sycl(D.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> e_sycl(E.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> f_sycl(F.data(), sycl::range<1>(array_size));

		madd2_async<float>(queue, &f_sycl, &d_sycl, 2.f, &e_sycl, 3.f, array_size, array_size, array_size);
	}
	for (unsigned int i = 0; i < array_size; i++) {
		if (F[i] != 2.f * D[i] + 3.f * E[i]) {
			std::cout << "The results are incorrect element " << i << " is " << F[i]
					  << " but inputs are " << D[i] << " and " << E[i] << "!\n";
			return 1;
		}
	}
	std::cout << "The results are correct for float!\n";
	return 0;
}
