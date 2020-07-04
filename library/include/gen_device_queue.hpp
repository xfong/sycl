#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#ifndef STRING__
#define STRING__
#include <string>
#include <cstring>
#endif // STRING__

#ifndef VECTOR__
#define VECTOR__
#include <vector>
#endif // VECTOR__

#ifndef GEN_DEVICE_QUEUE__
#define GEN_DEVICE_QUEUE__

// Utility function to get command line options for user
// to select GPU
int grabOpts(int nargs, char** argsList) {
	if (nargs < 3) {
		return -1;
	}
	for (int idx = 1; idx < nargs; idx++) {
		char* argField = argsList[idx];
		if (strcmp(argField, "-n") == 0) {
			return std::stoi(argsList[idx+1]);
		}
	}
	return -1;
}

// Utility function to enable user to select the GPU
// device to execute on. The input, targetDeviceNum,
// should be the index of the GPU found.
// If targetDeviceNum is -1 or greater than the
// available index, the default is to return the
// first GPU found
sycl::queue createSYCLqueue(const int targetDeviceNum) {
	// Grab all possible SYCL devices
	sycl::device targDevice;
	std::vector<sycl::device> deviceList;
	deviceList = targDevice.get_devices();
	std::vector<sycl::device> gpuList;
	// Go through all SYCL devices found and 
	// generate a list of the GPUs available
	for (int idx = 0; idx < deviceList.size(); idx++) {
		sycl::device currDevice = deviceList[idx];
		if (currDevice.is_gpu()) {
			gpuList.push_back(currDevice);
		}
	}
	if (gpuList.size() < 1) { // No GPUs found
		return sycl::queue();
	}
	int targIdx = targetDeviceNum;
	if ((targIdx >= gpuList.size()) || (targIdx < 0)) {
		targIdx = 0;
	}
	targDevice = gpuList[targIdx];
	return sycl::queue(targDevice);
}

#endif // GEN_DEVICE_QUEUE__
