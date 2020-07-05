#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#ifndef REDUCESUM_HPP_
#define REDUCESUM_HPP_

#ifndef VECTOR__
#define VECTOR__
#include <vector>
#endif // VECTOR__

#ifndef ARRAY__
#define ARRAY__
#include <array>
#endif // ARRAY__

#ifndef IOSTREAM__
#define IOSTREAM__
#include <iostream>
#endif // IOSTREAM__

template <typename dataT>
sycl::buffer<dataT> inline get_out_buffer(
    const size_t num_group, sycl::buffer<dataT> out_buffer) {
  return (num_group > 1)
             ? sycl::buffer<dataT>(sycl::range<1>{size_t(num_group)})
             : out_buffer;
}

template <typename dataT>
class reducesum_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;
	using local_accessor =
			sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::local>;
		reducesum_kernel(write_accessor dstPtr,
		             read_accessor srcPtr,
					 local_accessor localPtr,
					 dataT initVal,
					 size_t N)
		    :	dst(dstPtr),
				src(srcPtr),
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
			dataT thread_result = (global_thread_idx == 0) ? initVal : (dataT)(0); // Initialize initial values in all work-items

			// Coalesce access to global memory and track result in local register
			for (size_t gid = global_thread_idx; gid < N; gid += stride) {
				thread_result += src[gid];
			}

			// Initialize local memory before workgroup reduction
			size_t lid = item.get_local_linear_id();
			reduce_local_mem[lid] = thread_result;
			// Reduce workgroup data into the local memory
			for (size_t remaining_num = wgsize / 2; remaining_num > 0; remaining_num >>= 1) {
				// Ensure all threads have loaded their data into the local memory
				item.barrier(sycl::access::fence_space::local_space);
				// Execute work-item operation in binary tree reduction
				dataT myValue = (dataT)(0);
				if (lid < remaining_num) {
					myValue = reduce_local_mem[lid + remaining_num];
				}
				reduce_local_mem[lid] += myValue;
			}
			// Write reduced result to output for workgroup
			if  (lid == 0) {
				dst[item.get_group(0)] = reduce_local_mem[lid];
			}
		}
	private:
	    write_accessor dst;
	    read_accessor  src;
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
dataT reducesum_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *src,
                 dataT initVal,
                 size_t N,
				 const size_t gsize,
				 const size_t lsize) {

	using read_accessor_t =
		sycl::accessor<dataT, 1, sycl::access::mode::read,
					   sycl::access::target::global_buffer>;
	using write_accessor_t =
		sycl::accessor<dataT, 1, sycl::access::mode::write,
					   sycl::access::target::global_buffer>;

	using local_accessor_t =
		sycl::accessor<dataT, 1, sycl::access::mode::read_write,
				       sycl::access::target::local>;

	size_t num_group = lsize / lsize;

	dataT retVal;
	std::vector<dataT> out_data(1);
	sycl::buffer<dataT> out_buffer(out_data.data(), sycl::range<1>{1});
	out_buffer.set_final_data(nullptr);
	{

		// submitting the SYCL kernel to the cvengine SYCL queue.
		funcQueue.submit([&](sycl::handler &cgh) {
			// getting read access over the sycl buffer A inside the device kernel
			auto in_acc =
				src->template get_access<sycl::access::mode::read>(cgh);
			// getting write access over the sycl buffer C inside the device kernel
			auto out_acc =
				out_buffer.template get_access<sycl::access::mode::write>(cgh);

			auto local_acc = local_accessor_t(lsize, cgh);

			// constructing the kernel
			cgh.parallel_for(
				cl::sycl::nd_range<1>{sycl::range<1>{size_t(lsize)},
									  sycl::range<1>{size_t(lsize)}},
				reducesum_kernel<dataT>(out_acc, in_acc, local_acc, initVal, N));
		});
	}
	{
		auto h_acc = out_buffer.template get_access<sycl::access::mode::read>();
		retVal = h_acc[0];
	}
	return retVal;
}
#endif // REDUCESUM_HPP_