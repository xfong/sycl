// regionselect kernel

// device side function. This is essentially the function of the kernel
template <typename dataT>
void regionselect_fcn(sycl::nd_item<3> item,
                      dataT* dst,
                      dataT* src,
                      uint8_t* regions,
                      size_t N) {
    size_t gid = (item.get_group_id(1) * item.get_group_range(0) + item.get_group_id(0)) * item.get_local_range(0) + item.get_local_id(0);

    if (gid < N) {
        dstPtr[gid] = (regions[gid] == region ? src[gid] : (dataT)(0.0));
    }
}

template <typename dataT>
void regionselect_t(size_t blocks[3], size_t threads[3], sycl::queue q,
                 dataT* dst,
                 dataT* src,
                 uint8_t* regions,
                 size_t N) {
    q.parallel_for(sycl::nd_range<3>(sycl::range<3>(blocks[0]*threads[0], blocks[1]*threads[1], blocks[2]*threads[2]),
                                     sycl::range<3>(          threads[0],           threads[1],           threads[2])),
        [=](sycl::nd_item<3> item) {
            regionselect_fcn<dataT>(    dst,
                                        LUT,
                                    regions,
                                          N);
    });
}
