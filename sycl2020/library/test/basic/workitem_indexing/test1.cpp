#include "utils.h"

void simple_workitem_indexing(sycl::nd_item<3> item,
                     size_t* grpx, size_t* grpy, size_t* grpz,
                     size_t* thdx, size_t* thdy, size_t* thdz) {
    size_t gid = item.get_global_linear_id();
    thdx[gid] = item.get_global_id(0); // global thread id along x-dimension
    thdy[gid] = item.get_global_id(1); // global thread id along y-dimension
    thdz[gid] = item.get_global_id(2); // global thread id along z-dimension
    grpx[gid] = item.get_group(0);     // workgroup id along x-dimension for corresponding thread
    grpy[gid] = item.get_group(1);     // workgroup id along y-dimension for corresponding thread
    grpz[gid] = item.get_group(2);     // workgroup id along z-dimension for corresponding thread
}

int main() {
    auto queue = sycl::queue(sycl::gpu_selector());
    std::cout << "Working on device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "  Maximum number of Compute Units (CUs): " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "  Maximum work item dimensions: " << queue.get_device().get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
    auto result = queue.get_device().get_info<sycl::info::device::max_work_item_sizes>();
    std::cout << "  Maximum work item sizes: [" << result[0] << ", " << result[1] << ", " << result[2] << "]" << std::endl;
    std::cout << "  Maximum work group size: " << queue.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "  Preferred vector width (float): " << queue.get_device().get_info<sycl::info::device::preferred_vector_width_float>() << std::endl;
    std::cout << "  Preferred vector width (double): " << queue.get_device().get_info<sycl::info::device::preferred_vector_width_double>() << std::endl;
    std::cout << "  Native vector width (float): " << queue.get_device().get_info<sycl::info::device::native_vector_width_float>() << std::endl;
    std::cout << "  Native vector width (double): " << queue.get_device().get_info<sycl::info::device::native_vector_width_double>() << std::endl;
    std::cout << "  Maximum memory allocation size (bytes): " << queue.get_device().get_info<sycl::info::device::max_mem_alloc_size>() << std::endl;
    std::cout << "  Global memory size (bytes): " << queue.get_device().get_info<sycl::info::device::global_mem_size>() << std::endl;
    std::cout << "  Local memory size (bytes): " << queue.get_device().get_info<sycl::info::device::local_mem_size>() << std::endl;
    std::cout << "  Global cache memory size (bytes): " << queue.get_device().get_info<sycl::info::device::global_mem_cache_size>() << std::endl;
    std::cout << "  Global cache line size (bytes): " << queue.get_device().get_info<sycl::info::device::global_mem_cache_line_size>() << std::endl;

    size_t blocks[3] = {4, 2, 1};
    size_t threads[3] = {2, 4, 8};
    size_t total_size = blocks[0]*threads[0] * blocks[1]*threads[1] * blocks[2]*threads[2];

    auto grpx = static_cast<size_t*>(sycl::malloc_shared(total_size*sizeof(size_t), queue));
    auto grpy = static_cast<size_t*>(sycl::malloc_shared(total_size*sizeof(size_t), queue));
    auto grpz = static_cast<size_t*>(sycl::malloc_shared(total_size*sizeof(size_t), queue));
    auto thdx = static_cast<size_t*>(sycl::malloc_shared(total_size*sizeof(size_t), queue));
    auto thdy = static_cast<size_t*>(sycl::malloc_shared(total_size*sizeof(size_t), queue));
    auto thdz = static_cast<size_t*>(sycl::malloc_shared(total_size*sizeof(size_t), queue));
    queue.parallel_for(syclKernelLaunchGrid(blocks, threads),
        [=](sycl::nd_item<3> item){
            simple_workitem_indexing(item,
                                     grpx, grpy, grpz,
                                     thdx, thdy, thdz);
    });

    queue.wait();

    for (size_t idx = 0; idx < total_size; idx++) {
        std::cout << "    at index[" << idx << "]: grpx = " << grpx[idx] << "; grpy = " << grpy[idx] << "; grpz = " << grpz[idx] << ";" << std::endl;
        std::cout << "             " << idx << "]: thdx = " << thdx[idx] << "; thdy = " << thdy[idx] << "; thdz = " << thdz[idx] << ";" << std::endl;
    }
    sycl::free(grpx, queue);
    sycl::free(grpy, queue);
    sycl::free(grpz, queue);
    sycl::free(thdx, queue);
    sycl::free(thdy, queue);
    sycl::free(thdz, queue);
    return 0;
}
