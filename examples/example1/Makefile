CXX = g++
CXXFLAGS += -std=c++11 `pkg-config --cflags computecpp`
LDFLAGS += `pkg-config --libs computecpp`
SYCLFLAGS += `pkg-config computecpp --variable device_cflags`
SYCLCXX ?= `pkg-config computecpp --variable device_compiler`

EXECUTABLE = example1

all : $(EXECUTABLE)

%: %.cpp

%.o : %.cpp
	$(SYCLCXX) $(SYCLFLAGS) $< -o $@.sycl
	$(CXX) -c $(CXXFLAGS) -include $@.sycl $< -o $@

% : %.o
	$(CXX) $< $(LDFLAGS) -o $@

.PHONY: clean help

clean:
	rm -f *.o *.sycl $(EXECUTABLE)
