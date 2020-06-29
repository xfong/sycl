CXX = g++
CXXFLAGS += -std=c++11 -I$(COMPUTECPP_ROOT_DIR)/include -I$(OCL_PATH)/include
LDFLAGS += -L$(COMPUTECPP_ROOT_DIR)/lib -lComputeCpp
SYCLFLAGS += -I$(COMPUTECPP_ROOT_DIR)/include -I$(OCL_PATH)/include -O2 -mllvm -inline-threshold=1000 -sycl -sycl-target ptx64
SYCLCXX ?= compute++

EXECUTABLE = simple_vec_add

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
