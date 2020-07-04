/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  custom-device-selector.cpp
 *
 *  Description:
 *    Sample code that shows how to write a custom device selector in SYCL.
 *
 **************************************************************************/

#include <iostream>
#include <string>
#include <cstring>

#include <CL/sycl.hpp>

using namespace cl::sycl;

int setup_opts(int nargs, char** argsList) {
	if (nargs < 3) {
		return -1;
	}
	for (int idx = 1; idx < nargs; idx++) {
		char* argField = argsList[idx];
		if (strcmp(argField, "-n") == 0) {
			return std::stoi(argsList[idx+1]);
		}
	}
}

int main(int argc, char** argv) {
	int dev_num = setup_opts(argc, argv);
	if (dev_num < 0) {
		std::cout << "No proper arguments were found!" << std::endl;
		return -1;
	}
	std::cout << "Targeting GPU" << dev_num << "..." << std::endl;
	cl_int err;
	sycl::device targDevice;
	std::vector<sycl::device> deviceList;
	deviceList = targDevice.get_devices();
	//err = cl::get(platformList);
	std::cout << "Found " << deviceList.size() << " number of devices" << std::endl;
	std::vector<sycl::device> gpuList;
	for (int idx = 0; idx < deviceList.size(); idx++) {
		sycl::device currDevice = deviceList[idx];
		if (currDevice.is_gpu()) {
			gpuList.push_back(currDevice);
		}
	}
	if (gpuList.size() < 1) {
		std::cout << "No GPU device found!" << std::endl;
		return -1;
	}
	std::cout << "Found " << gpuList.size() << " number of GPUs" << std::endl;
	if (dev_num > gpuList.size() - 1) {
		dev_num = 0;
	}
	targDevice = gpuList[dev_num];

	std::cout << "Setting up queue on device:" << targDevice.get_info<sycl::info::device::name>();
	sycl::queue myQueue(targDevice);
	return 0;
}