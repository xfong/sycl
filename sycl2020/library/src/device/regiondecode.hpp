// regiondecode kernel

// device side function. This is essentially the function of the kernel
// decode the regions+LUT pair into an uncompressed array
template <typename dataT>
void regiondecode_fcn(sycl::nd_item<3> item,
                      dataT* dst,
                      dataT* LUT,
                      uint8_t* regions,
                      size_t N) {
    size_t gid = (item.get_group(1) * item.get_group_range(0) + item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);

    if (gid < N) {
        dstPtr[gid] = LUT[regions[gid]];
    }
}

// the function that launches the kernel
template <typename dataT>
void regiondecode_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                    dataT* dst,
                    dataT* LUT,
                    uint8_t* regions,
                    size_t N) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item) {
            regiondecode_fcn<dataT>(    dst,
                                        LUT,
                                    regions,
                                          N);
    });
}
