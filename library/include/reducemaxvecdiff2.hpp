#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#ifndef ARRAY__
#define ARRAY__
#include <array>
#endif // ARRAY__

#include "reducemaxabs.hpp"

template <typename dataT>
class reducemaxvecdiff2_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using local_accessor =
			sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::local>;
		reducemaxvecdiff2_kernel(write_accessor dstPtr,
		             read_accessor src0xPtr, read_accessor src0yPtr, read_accessor src0zPtr,
		             read_accessor src1xPtr, read_accessor src1yPtr, read_accessor src1zPtr,
					 local_accessor localPtr,
					 dataT initVal,
					 size_t N)
		    :	dst(dstPtr),
				src0x(src0xPtr),
				src0y(src0yPtr),
				src0z(src0zPtr),
				src1x(src1xPtr),
				src1y(src1yPtr),
				src1z(src1zPtr),
				reduce_local_mem(localPtr),
				initVal(initVal),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t wgsize = item.get_local_range(0);
			// Exit if all work-items in the group will not execute
			if (item.get_group(0)*wgsize >= N ) {
				return;
			}

			size_t stride = item.get_global_range(0);
			size_t global_thread_idx = item.get_global_linear_id();
			dataT thread_result = initVal; // Initialize initial values in all work-items

			// Coalesce access to global memory and track result in local register
			for (size_t gid = global_thread_idx; gid < N; gid += stride) {
				// x-component
				dataT val0 = src0x[gid];
				dataT val1 = src1x[gid];
				dataT dVal = val0 - val1;
				dataT element = dVal*dVal;

				// y-component
				val0 = src0y[gid];
				val1 = src1y[gid];
				dVal = val0 - val1;
				element += dVal*dVal;

				// z-component
				val0 = src0z[gid];
				val1 = src1z[gid];
				dVal = val0 - val1;
				element += dVal*dVal;

				thread_result = sycl::fmax(thread_result, element);
			}

			// Initialize local memory before workgroup reduction
			size_t lid = item.get_local_linear_id();
			reduce_local_mem[lid] = thread_result;
			// Reduce workgroup data into the local memory
			for (size_t remaining_num = wgsize / 2; remaining_num > 0; remaining_num >>= 1) {
				// Ensure all threads have loaded their data into the local memory
				item.barrier(sycl::access::fence_space::local_space);
				// Execute work-item operation in binary tree reduction
				if (lid < remaining_num) {
					dataT other = reduce_local_mem[lid + remaining_num];
					dataT mine = reduce_local_mem[lid];
					reduce_local_mem[lid] = fmax(mine, other);
				}
			}
			// Write reduced result to output
			if  (lid == 0) {
				dst[item.get_group(0)] = reduce_local_mem[lid];
			}
		}
	private:
	    write_accessor dst;
	    read_accessor  src0x;
	    read_accessor  src0y;
	    read_accessor  src0z;
	    read_accessor  src1x;
	    read_accessor  src1y;
	    read_accessor  src1z;
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
dataT reducemaxvecdiff2_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *x1,
                 sycl::buffer<dataT, 1> *y1,
                 sycl::buffer<dataT, 1> *z1,
                 sycl::buffer<dataT, 1> *x2,
                 sycl::buffer<dataT, 1> *y2,
                 sycl::buffer<dataT, 1> *z2,
                 dataT initVal,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

	using local_accessor_t =
		sycl::accessor<dataT, 1, sycl::access::mode::read_write,
				       sycl::access::target::local>;

	auto num_groups = gsize / lsize;

	dataT retVal;
	std::vector<dataT> out_data(1);
	sycl::buffer<dataT> out_buffer(out_data.data(), sycl::range<1>{1});
	out_buffer.set_final_data(nullptr);
	{
		// First stage in reduction with maximum number of work-groups
		// Each work-group will output a result
		sycl::buffer<dataT> temp_buffer = get_out_buffer(num_groups, out_buffer);

		// First stage in reduction with maximum number of work-groups
		// Each work-group will output a result
		funcQueue.submit([&](sycl::handler &cgh) {
			// getting read access over the input sycl buffers inside the device kernel
			auto x1_acc =
				x1->template get_access<sycl::access::mode::read>(cgh);
			auto y1_acc =
				y1->template get_access<sycl::access::mode::read>(cgh);
			auto z1_acc =
				z1->template get_access<sycl::access::mode::read>(cgh);
			auto x2_acc =
				x2->template get_access<sycl::access::mode::read>(cgh);
			auto y2_acc =
				y2->template get_access<sycl::access::mode::read>(cgh);
			auto z2_acc =
				z2->template get_access<sycl::access::mode::read>(cgh);
			// getting write access over the output sycl buffer inside the device kernel
			auto temp_acc =
				temp_buffer.template get_access<sycl::access::mode::write>(cgh);
			// getting read/write access over the local buffer inside the device kernel
			auto local_acc = local_accessor_t(lsize, cgh);

			// constructing the kernel
			cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{size_t(gsize)},
											   sycl::range<1>{size_t(lsize)}},
							 reducemaxvecdiff2_kernel<dataT>(temp_acc, x1_acc, y1_acc, z1_acc, x2_acc, y2_acc, z2_acc, reduce_local_mem, initVal, N));
		});
		// Second stage in reduction with one work-group
		// Executed only if the first stage consists of several work-groups
		if (num_groups > 1) {
			funcQueue.submit([&](sycl::handler &cgh) {
				// getting read access over the input sycl buffer inside the device kernel
				auto in_acc =
					temp_buffer.template get_access<sycl::access::mode::read>(cgh);
				// getting write access over the output sycl buffer inside the device kernel
				auto temp_acc =
					out_buffer.template get_access<sycl::access::mode::write>(cgh);
				// getting read/write access over the local buffer inside the device kernel
				auto local_acc = local_accessor_t(lsize, cgh);

				// constructing the kernel
				cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{size_t(lsize)},
												   sycl::range<1>{size_t(lsize)}},
								 reducemaxabs_kernel<dataT>(temp_acc, input_acc, local_acc, dataT(0), num_groups));
			});
		}
	}
	{
		auto h_acc = out_buffer.template get_access<sycl::access::mode::read>();
		retVal = h_acc[0];
	}
	return retVal;
}
