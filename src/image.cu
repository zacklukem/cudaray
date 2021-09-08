#include "image.h"

#include "glm.inl"
#include <cstdint>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace cudaray;

__host__ Image::Image() {}

__host__ __device__ Image::Image(glm::vec3* data, size_t width, size_t height)
    : data(data), width(width), height(height) {}

__device__ Image::Image(size_t width, size_t height)
    : width(width), height(height) {
  cudaMalloc(&data, width * height * sizeof(glm::vec3));
}

__host__ __device__ Image::~Image() {
  // #ifdef __CUDA_ARCH__
  //   cudaFree(this->data);
  // #else
  //   free(this->data);
  // #endif
}

__device__ const glm::vec3& Image::getPixel(int x, int y) const {
  return this->data[x + y * width];
}

__device__ const glm::vec3& Image::getPixel(const glm::ivec2& pos) const {
  return getPixel(pos.x, pos.y);
}

__device__ const glm::vec3& Image::getPixel(const glm::vec2& pos) const {
  auto x = (int)(pos.x * ((float)width - 1.0f));
  auto y = (int)(pos.y * ((float)height - 1.0f));
  return getPixel(x, y);
}

__device__ void Image::setPixel(int x, int y, const glm::vec3& value) {
  this->data[x + y * width] = value;
}

__device__ void Image::setPixel(const glm::ivec2& pos, const glm::vec3& value) {
  setPixel(pos.x, pos.y, value);
}

__device__ void Image::setPixel(const glm::vec2& pos, const glm::vec3& value) {
  auto x = (int)(pos.x * ((float)width - 1.0f));
  auto y = (int)(pos.y * ((float)height - 1.0f));
  setPixel(x, y, value);
}

__host__ void Image::writeToFile(const char* filename) const {
  uint8_t(*byte_data)[3] = (uint8_t(*)[3])malloc(width * height * 3);

  for (int i = 0; i < width * height; i++) {
    auto pixel = data[i];
    byte_data[i][0] =
        static_cast<uint8_t>(glm::clamp(pixel.x, 0.0f, 1.0f) * 255.0f + 0.5f);
    byte_data[i][1] =
        static_cast<uint8_t>(glm::clamp(pixel.y, 0.0f, 1.0f) * 255.0f + 0.5f);
    byte_data[i][2] =
        static_cast<uint8_t>(glm::clamp(pixel.z, 0.0f, 1.0f) * 255.0f + 0.5f);
  }

  stbi_write_png(filename, width, height, 3, (void*)byte_data, width * 3);
  free(byte_data);
}

__device__ glm::vec2 Image::getUVCoord(const glm::ivec2& pos) const {
  return glm::vec2((float)pos.x / ((float)width - 1.0f),
                   (float)pos.y / ((float)height - 1.0f));
}

__device__ glm::vec2 Image::getUVCoord(const glm::ivec2& pos,
                                       const glm::vec2& rand) const {
  return glm::vec2(((float)pos.x + rand.x) / ((float)width - 1.0f),
                   ((float)pos.y + rand.y) / ((float)height - 1.0f));
}