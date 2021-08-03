#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 4;

int main(){
  queue q(property::queue::in_order{});

  int *data = malloc_shared<int>(N, q);
  for(int i=0; i<N; i++) data[i] = 10;

  q.parallel_for(range<1>(N), [=] (id<1> i){
    data[i] += 2;
  });

  q.parallel_for(range<1>(N), [=] (id<1> i){
    data[i] += 3;
  }).wait();

  for(int i=0; i<N; i++) std::cout << data[i] << std::endl;
  free(data, q);
  return 0;
}

