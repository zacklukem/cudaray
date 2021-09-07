#ifndef SRC_CUDA
#define SRC_CUDA

#include <iostream>

namespace cudaray {

template <class T> class DevicePtr {

public:
  __host__ DevicePtr(T* dev_ptr) : ptr(dev_ptr){};
  __host__ ~DevicePtr() { cudaFree(ptr); };

  // __host__ DevicePtr(const DevicePtr& obj) = delete;
  // __host__ DevicePtr& operator=(const DevicePtr& obj) = delete;

  __host__ T* operator*() { return this->ptr; };

private:
  T* ptr;
};

class CudaObject {

public:
  template <class T> __host__ DevicePtr<T> asDeviceObject() {
    return DevicePtr<T>((T*)moveToDevice());
  };

protected:
  virtual __host__ void* moveToDevice() = 0;
};

namespace __internal__ {
template <class T, class... Args>
__global__ void construct_object(T** object_pointer, Args... args) {
  if (threadIdx.x == 0 && blockIdx.x == 0) { // Only construct max of 1
    *object_pointer = new T(args...);
  }
}

template <class T> __global__ void deconstruct_object(T* object_pointer) {
  delete object_pointer;
}
} // namespace __internal__

template <class T, class... Args> __host__ T* make_in_device(Args... args) {
  T** ptr_ptr;
  cudaMalloc(&ptr_ptr, sizeof(T));
  __internal__::construct_object<T><<<1, 1>>>(ptr_ptr, args...);
  T* out;
  cudaMemcpy(&out, ptr_ptr, sizeof(T*), cudaMemcpyDeviceToHost);
  cudaFree(ptr_ptr);
  return out;
}

template <class T> __host__ void free_in_device(T* ptr) {
  __internal__::deconstruct_object<T><<<1, 1>>>(ptr);
}

} // namespace cudaray

#endif // SRC_CUDA
