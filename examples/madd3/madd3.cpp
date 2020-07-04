#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "gen_device_queue.hpp"
#include "sycl_engine.hpp"

#ifndef VECTOR__
#define VECTOR__
#include <vector>
#endif // VECTOR__

#ifndef IOSTREAM__
#define IOSTREAM__
#include <iostream>
#endif // IOSTREAM__

int main(int argc, char** argv) {
	int gpu_num = grabOpts(argc, argv);

	const size_t array_size = 2048;
	std::vector<int> A(array_size), B(array_size), C(array_size), D(array_size);
	
	std::fill(A.begin(), A.end(), 1);
	std::fill(B.begin(), B.end(), 3);
	std::fill(C.begin(), C.end(), 150);
	std::fill(D.begin(), D.end(), 100000);

	std::vector<float> E(array_size), F(array_size), G(array_size), H(array_size);
	
	std::fill(E.begin(), E.end(), 0.01f);
	std::fill(F.begin(), F.end(), 1.0f);
	std::fill(G.begin(), G.end(), 10.0f);
	std::fill(H.begin(), H.end(), 100000.0f);

	// Need to select the OpenCL device to use first
	sycl::default_selector device_selector;

	// Then, set up command queue on OpenCL device
	sycl::queue queue = createSYCLqueue(gpu_num);
	std::cout << "Executing on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

	// Create memory buffers that the OpenCL will access
	// The template is cl::sycl:buffer<type, dims>
	// The constructor takes a host pointer to the host memory location where
	// associated buffer data is held, and the number of elements
	{
		sycl::buffer<sycl::cl_int, 1> a_sycl(A.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_int, 1> b_sycl(B.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_int, 1> c_sycl(C.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_int, 1> d_sycl(D.data(), sycl::range<1>(array_size));

		madd3_async<int>(queue, &d_sycl, &a_sycl, 2, &b_sycl, 3, &c_sycl, 5, array_size, array_size / 4, array_size / 8);
	}
	for (unsigned int i = 0; i < array_size; i++) {
		if (D[i] != 2 * A[i] + 3 * B[i] + 5 * C[i]) {
			std::cout << "The results are incorrect element " << i << " is " << D[i]
					  << " but inputs are " << A[i] << ", " << B[i] << " and " << C[i] << "!\n";
			return 1;
		}
	}
	std::cout << "The results are correct for int!\n";

	// Create memory buffers that the OpenCL will access
	// The template is cl::sycl:buffer<type, dims>
	// The constructor takes a host pointer to the host memory location where
	// associated buffer data is held, and the number of elements
	{
		sycl::buffer<sycl::cl_float, 1> e_sycl(E.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> f_sycl(F.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> g_sycl(G.data(), sycl::range<1>(array_size));
		sycl::buffer<sycl::cl_float, 1> h_sycl(H.data(), sycl::range<1>(array_size));

		madd3_async<float>(queue, &h_sycl, &e_sycl, 2.f, &f_sycl, 3.f, &g_sycl, 5.f, array_size, array_size / 4, array_size / 8);
	}
	for (unsigned int i = 0; i < array_size; i++) {
		if (H[i] != 2.f * E[i] + 3.f * F[i] + 5.f * G[i]) {
			std::cout << "The results are incorrect element " << i << " is " << H[i]
					  << " but inputs are " << E[i] << ", " << F[i] << " and " << G[i] << "!\n";
			return 1;
		}
	}
	std::cout << "The results are correct for float!\n";
	return 0;
}
