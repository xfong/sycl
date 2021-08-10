// regionadds kernel

// device side function. This is essentially the function of the kernel
// add region-based scalar to dst:
// dst[i] += LUT[region[i]]
template <typename dataT>
void regionadds_fcn(sycl::nd_item<3> item,
                    dataT* dst,
                    dataT* LUT,
                    uint8_t* regions,
                    size_t N) {
    size_t gid = (item.get_group(1) * item.get_group_range(0) + item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);

    if (gid < N) {
        uint8_t r = regions[gid];
        dst[gid] += LUT[r];
    }
}

// the function that launches the kernel
template <typename dataT>
void regionadds_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                  dataT* dst,
                  dataT* LUT,
                  uint8_t* regions,
                  size_t N) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item) {
            regionadds_fcn<dataT>(    dst,
                                      LUT,
                                  regions,
                                        N);
    });
}
