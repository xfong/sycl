// Define a special buffer structure that has a host side pointer
// and a device side buffer. This can allow SYCL runtimes to handle
// data movement between host and device. But we need to define
// different data types to facilitate kernels

#include <CL/sycl.hpp>

// templated class
template<typename dataT>
class syclDataType_t {
    public:
        syclDataType_t(size_t nb) {
            this->num_elements = nb;
            this->num_bytes = nb * sizeof(dataT);
            this->hostPtr = (dataT*) malloc(this->num_bytes);
            this->buf = new cl::sycl::buffer<dataT, 1>(this->hostPtr, cl::sycl::range<1>(nb));
        }
        syclDataType_t(size_t nb, bool host) {
            this->num_elements = nb;
            this->num_bytes = nb * sizeof(dataT);
            if (host) {
                this->hostPtr = (dataT*) malloc(this->num_bytes);
                this->buf = new cl::sycl::buffer<dataT, 1>(this->hostPtr, cl::sycl::range<1>(nb));
            } else {
                this->hostPtr =  NULL;
                this->buf = new cl::sycl::buffer<dataT, 1>(cl::sycl::range<1>(nb));
            }
        }
        ~syclDataType_t() {
            free(this->buf);
            if (this->hostPtr != NULL) {
                free(this->hostPtr);
            }
        }
        cl::sycl::buffer<dataT, 1>* getBuffer() { return this->buf; }
        dataT* getPointer() { return this->hostPtr; }
        size_t size() { return this->num_elements; }
        size_t bytes() { return this->num_bytes; }
    private:
        dataT* hostPtr;
        cl::sycl::buffer<dataT, 1>* buf;
        size_t num_bytes;
        size_t num_elements;
};

// Boolean type
typedef syclDataType_t<bool>                    syclBool_t;

// Char type
typedef syclDataType_t<char>                    syclChar_t;
typedef syclDataType_t<wchar_t>                 syclWChar_t;
typedef syclDataType_t<char16_t>                syclChar16_t;
typedef syclDataType_t<char32_t>                syclChar32_t;

// Signed integer types
typedef syclDataType_t<short>                   syclShort_t;
typedef syclDataType_t<signed short>            syclSShort_t;
typedef syclDataType_t<short int>               syclShortInt_t;
typedef syclDataType_t<signed short int>        syclSShortInt_t;
typedef syclDataType_t<int>                     syclInt_t;
typedef syclDataType_t<signed int>              syclSInt_t;
typedef syclDataType_t<signed>                  syclSigned_t;
typedef syclDataType_t<long>                    syclLong_t;
typedef syclDataType_t<long int>                syclLongInt_t;
typedef syclDataType_t<signed long>             syclSLong_t;
typedef syclDataType_t<signed long int>         syclSLongInt_t;
typedef syclDataType_t<long long>               syclLLong_t;
typedef syclDataType_t<long long int>           syclLLongInt_t;
typedef syclDataType_t<signed long long>        syclSLLong_t;
typedef syclDataType_t<signed long long int>    syclSLLongInt_t;
typedef syclDataType_t<int8_t>                  syclInt8_t;
typedef syclDataType_t<int16_t>                 syclInt16_t;
typedef syclDataType_t<int32_t>                 syclInt32_t;
typedef syclDataType_t<int64_t>                 syclInt64_t;

// Unsigned integer types
typedef syclDataType_t<unsigned short>          syclUShort_t;
typedef syclDataType_t<unsigned short int>      syclUShortInt_t;
typedef syclDataType_t<unsigned int>            syclUInt_t;
typedef syclDataType_t<unsigned>                syclUnsigned_t;
typedef syclDataType_t<unsigned long>           syclULong_t;
typedef syclDataType_t<unsigned long int>       syclULongInt_t;
typedef syclDataType_t<unsigned long long>      syclULLong_t;
typedef syclDataType_t<unsigned long long int>  syclULLongInt_t;
typedef syclDataType_t<uint8_t>                 syclUint8_t;
typedef syclDataType_t<uint16_t>                syclUint16_t;
typedef syclDataType_t<uint32_t>                syclUint32_t;
typedef syclDataType_t<uint64_t>                syclUint64_t;

// Floating point types
typedef syclDataType_t<float>                   syclFloat_t;
typedef syclDataType_t<double>                  syclDouble_t;
typedef syclDataType_t<long double>             syclLDouble_t;
