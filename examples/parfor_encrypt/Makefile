CXX = g++
CXXFLAGS += -std=c++11 -I/opt/ComputeCpp-CE-2.0.0-x86_64-linux-gnu/include
LDFLAGS += -L/opt/ComputeCpp-CE-2.0.0-x86_64-linux-gnu/lib -lComputeCpp
SYCLFLAGS += -I/opt/ComputeCpp-CE-2.0.0-x86_64-linux-gnu/include -O2 -mllvm -inline-threshold=1000 -sycl -sycl-target ptx64
SYCLCXX ?= compute++

EXECUTABLE = parfor_encrypt

all : $(EXECUTABLE)

%: %.cpp

%.o : %.cpp
	$(SYCLCXX) $(SYCLFLAGS) $(SYCLTARGS) $< -o $@.sycl
	$(CXX) -c $(CXXFLAGS) -include $@.sycl $< -o $@

% : %.o
	$(CXX) $< $(LDFLAGS) -o $@

.PHONY: clean help

clean:
	rm -f *.o *.sycl $(EXECUTABLE)
