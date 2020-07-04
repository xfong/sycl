#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "gen_device_queue.hpp"

#ifndef CSTDLIB__
#define CSTDLIB__
#include <cstdlib>
#endif // CSTDLIB__

#ifndef CTIME__
#define CTIME__
#include <ctime>
#endif // CTIME__

#ifndef VECTOR__
#define VECTOR__
#include <vector>
#endif // VECTOR__

#ifndef IOSTREAM__
#define IOSTREAM__
#include <iostream>
#endif // IOSTREAM__

template <typename dataT>
class dotproduct_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		dotproduct_kernel(write_accessor dstPtr,
					 dataT prefactor,
		             read_accessor a0Ptr,
					 read_accessor a1Ptr,
					 read_accessor a2Ptr,
		             read_accessor b0Ptr,
					 read_accessor b1Ptr,
					 read_accessor b2Ptr,
					 size_t N)
		    :	dstPtr(dstPtr),
				prefactor(prefactor),
				axPtr(a0Ptr),
				ayPtr(a1Ptr),
				azPtr(a2Ptr),
				bxPtr(b0Ptr),
				byPtr(b1Ptr),
				bzPtr(b2Ptr),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				sycl::vec<dataT, 3> aVec = { axPtr[gid], ayPtr[gid], azPtr[gid] };
				sycl::vec<dataT, 3> bVec = { bxPtr[gid], byPtr[gid], bzPtr[gid] };
				dstPtr[gid] += prefactor * sycl::dot(aVec, bVec);
			}
		}
	private:
	    write_accessor dstPtr;
		dataT          prefactor;
	    read_accessor  axPtr;
	    read_accessor  ayPtr;
	    read_accessor  azPtr;
	    read_accessor  bxPtr;
	    read_accessor  byPtr;
	    read_accessor  bzPtr;
		size_t         N;
};

template <typename dataT>
void dotproduct_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *dst,
                 dataT prefactor,
                 sycl::buffer<dataT, 1> *a0,
                 sycl::buffer<dataT, 1> *a1,
                 sycl::buffer<dataT, 1> *a2,
                 sycl::buffer<dataT, 1> *b0,
                 sycl::buffer<dataT, 1> *b1,
                 sycl::buffer<dataT, 1> *b2,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto dst_acc = dst->template get_access<sycl::access::mode::read_write>(cgh);
        auto a0_acc = a0->template get_access<sycl::access::mode::read>(cgh);
        auto a1_acc = a1->template get_access<sycl::access::mode::read>(cgh);
        auto a2_acc = a2->template get_access<sycl::access::mode::read>(cgh);
        auto b0_acc = b0->template get_access<sycl::access::mode::read>(cgh);
        auto b1_acc = b1->template get_access<sycl::access::mode::read>(cgh);
        auto b2_acc = b2->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 dotproduct_kernel<dataT>(dst_acc, prefactor, a0_acc, a1_acc, a2_acc, b0_acc, b1_acc, b2_acc, N));
    });
}

int main(int argc, char** argv) {

	int gpu_num = grabOpts(argc, argv);
	
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
		G[idx] = 0.0f;
	}

	// Then, set up command queue on OpenCL device
	sycl::queue queue = createSYCLqueue(gpu_num);

	std::cout << "Executing on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

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

		dotproduct_async<cl_float>(queue, &g_sycl, 1.f, &a_sycl, &b_sycl, &c_sycl, &d_sycl, &e_sycl, &f_sycl, array_size, 1024, 256);
		queue.wait();
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
