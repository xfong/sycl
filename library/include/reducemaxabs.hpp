#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#ifndef REDUCEMAXABS__
#define REDUCEMAXABS__
#ifndef ARRAY__
#define ARRAY__
#include <array>
#endif // ARRAY__

template <typename dataT>
class reducemaxabs_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		using local_accessor =
			sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::local>;
		reducemaxabs_kernel(write_accessor dstPtr,
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
			dataT thread_result = initVal; // Initialize initial values in all work-items

			// Coalesce access to global memory and track result in local register
			for (size_t gid = global_thread_idx; gid < N; gid += stride) {
				dataT element = sycl::fabs(src[gid]);
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
dataT reducemaxabs_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *src,
                 dataT initVal,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

	using local_accessor_t =
		sycl::accessor<dataT, 1, sycl::access::mode::read_write,
				       sycl::access::target::local>;

	size_t num_groups = gsize / lsize;

	dataT retVal;
	std::vector<dataT> out_data(1);
	sycl::buffer<dataT> out_buffer(out_data.data(), sycl::range<1>{1});
	out_buffer.set_final_data(nullptr);

	{

		sycl::buffer<dataT> temp_buffer = get_out_buffer(num_groups, out_buffer);
		
		// First stage in reduction with maximum number of work-groups
		// Each work-group will output a result
		funcQueue.submit([&] (sycl::handler& cgh) {
			// getting read access over the input sycl buffer inside the device kernel
			auto in_acc =
				src->template get_access<sycl::access::mode::read>(cgh);
			// getting write access over the output sycl buffer inside the device kernel
			auto temp_acc =
				temp_buffer.template get_access<sycl::access::mode::write>(cgh);
			// getting read/write access over the local buffer inside the device kernel
			auto local_acc = local_accessor_t(lsize, cgh);

			// constructing the kernel
			cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{size_t(gsize)},
											   sycl::range<1>{size_t(lsize)}},
							 reducemaxabs_kernel<dataT>(temp_acc, input_acc, local_acc, initVal, N));
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
#endif // REDUCEMAXABS__