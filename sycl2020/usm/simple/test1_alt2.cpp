#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 16;

void kernel_call(queue q, int count, int* input) {
  q.submit([&] (handler& cgh) {
    cgh.parallel_for<class scale_kernel>(range<1>(count), [=](id<1> i){
      input[i] *= 2;
    });
  });
  q.wait();
}

int main(){
  queue q;

  int *data = malloc_shared<int>(N, q);
  for(int i=0; i<N; i++) data[i] = i;

  kernel_call(q, N, data);

  for(int i=0; i<N; i++) std::cout << data[i] << std::endl;
  free(data, q);
  return 0;
}

