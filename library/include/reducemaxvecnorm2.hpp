#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#ifndef ARRAY_INC__
#define ARRAY_INC__
#include <array>
#endif // ARRAY_INC__

#include "reducemaxabs.hpp"
#ifndef MATH_HPP__
#define <cmath>
#endif // MATH_HPP__

template <typename dataT>
class reducemaxvecnorm2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
	using local_accessor =
			sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::local>;
		reducemaxvecnorm2_kernel(write_accessor dstPtr,
		             read_accessor src0xPtr, read_accessor src0yPtr, read_accessor src0zPtr,
					 local_accessor localPtr,
					 dataT initVal,
					 size_t N)
		    :	dst(dstPtr),
				src0x(src0xPtr),
				src0y(src0yPtr),
				src0z(src0zPtr),
				reduce_local_mem(localPtr),
				initVal(initVal),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			// Exit if all work-items in the group will not execute
			if (item.get_group(0)*item.get_local_range(0) >= N ) {
				return;
			}

			size_t stride = item.get_global_range(0);
			auto global_thread_idx = item.get_global_linear_id();
			dataT thread_result = initVal; // Initialize initial values in all work-items

			// Coalesce access to global memory and track result in local register
			for (size_t gid = global_thread_idx; gid < N; gid += stride) {
				// x-component
				dataT val0 = src0x[gid];
				dataT element = val0*val0;

				// y-component
				val0 = src0y[gid];
				element += val0*val0;

				// y-component
				val0 = src0z[gid];
				element += val0*val0;

				thread_result = sycl::fmax(thread_result, element);
			}

			// Initialize local memory before workgroup reduction
			reduce_local_mem[global_thread_idx] = thread_result;
			// Reduce workgroup data into the local memory
			for (size_t remaining_num = item.get_local_range(0) / 2; remaining_num > 0; remaining_num >> 1) {
				// Ensure all threads have loaded their data into the local memory
				item.barrier(sycl::access::fence_space::local_space);
				auto lid = item.get_local_linear_id();
				// Execute work-item operation in binary tree reduction
				if (lid < remaining_num) {
					dataT other = reduce_local_mem[lid + remaining_num];
					dataT mine = reduce_local_mem[lid];
					reduce_local_mem[lid] = fmax(mine, other);
				}
			}
			// Write reduced result to output
			if  (lid == 0) {
				dst[item.get_group_linear_id()] = reduce_local_mem[0];
			}
		}
	private:
	    write_accessor dst;
	    read_accessor  src0x;
	    read_accessor  src0y;
	    read_accessor  src0z;
		local_accessor reduce_local_mem;
		dataT          initVal;
		size_t         N;
};

// Main function call that will call the kernels internally
// the reduction function will work in two stages
//     The first stage performs reduction until the result can be
//     obtained using one workgroup
//     The final stage will use one workgroup to perform the
//     final reduction
template <typename dataT>
dataT reducemaxvecnorm2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *x1,
                 sycl::buffer<dataT, 1> *y1,
                 sycl::buffer<dataT, 1> *z1,
                 dataT initVal,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

	auto int_size = gsize / lsize;
	size_t totalThreads = gsize;
	// First stage in reduction with maximum number of work-groups
	// Each work-group will output a result
	if (int_size > lsize) {
		int_size = lsize;
		totalThreads = lsize*lsize;
	}
	std::array<dataT, int_size> int_result;
	auto intBuffer = sycl::buffer<dataT, 1>(int_result.data(), int_size);
	funcQueue.submit([&] (sycl::handler& cgh) {
		auto x1_acc = x1->template get_access<sycl::access::mode::read>(cgh);
		auto y1_acc = y1->template get_access<sycl::access::mode::read>(cgh);
		auto z1_acc = z1->template get_access<sycl::access::mode::read>(cgh);
		auto output_acc = intBuffer.get_access<sycl::access::mode::read_write>(cgh);
		sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::local> reduce_local_mem(sycl::range<1>(lsize), cgh);

		cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(totalThreads),
										   sycl::range<1>(lsize)),
						 reducemaxvecnorm2_kernel<dataT>(output_acc, x1_acc, y1_acc, z1_acc, reduce_local_mem, dataT(0), N));
	}
	// If first stage produces final reduced result, we will return output result
	if (int_size == 1) {
		return std::sqrt(int_result[0]) + initVal;
	}
	// Second stage in reduction with one work-group
	// Acts on results of stage one to produce final result
	std::array<dataT, 1> out_result;
	auto outBuffer = sycl::buffer<dataT, 1>(out_result.data(), 1);
	funcQueue.submit([&] (sycl::handler& cgh) {
		auto input_acc = intBuffer.get_access<sycl::access::mode::read>(cgh);
		auto output_acc = outBuffer.get_access<sycl::access::mode::read_write>(cgh);
		sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::local> reduce_local_mem(sycl::range<1>(lsize), cgh);

		cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(lsize),
										   sycl::range<1>(lsize)),
						 reducemaxabs_kernel<dataT>(output_acc, input_acc, reduce_local_mem, dataT(0), N));
	}
	return std::sqrt(out_result[0]) + initVal;
}
