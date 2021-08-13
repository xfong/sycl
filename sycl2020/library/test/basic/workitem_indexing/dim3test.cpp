#include "utils.h"

int main() {
    dim3 test;
    test = dim3(64);
    std::cout << "1. test: " << test.x << ", " << test.y << ", " << test.z << std::endl;
    test = dim3(4, 2);
    std::cout << "2. test: " << test.x << ", " << test.y << ", " << test.z << std::endl;
    test = dim3(8, 16, 32);
    std::cout << "3. test: " << test.x << ", " << test.y << ", " << test.z << std::endl;
    return 0;
}
