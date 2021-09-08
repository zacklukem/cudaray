#ifndef IMAGE_H
#define IMAGE_H

#include "cuda.h"
#include "glm.inl"
#include <cstdint>

namespace cudaray {

/// Class representing an image
///
class Image {
public:
  __host__ Image();
  /// Construct a new instance of the Image class with the given width and
  /// height and allocate the required memory to store the internal image data
  __device__ Image(size_t width, size_t height);
  __host__ __device__ Image(glm::vec3* data, size_t width, size_t height);
  __host__ __device__ ~Image();

  /// Get the value of a pixel at coordinates x, y
  ///
  __device__ const glm::vec3& getPixel(int x, int y) const;

  /// Get the value of a pixel at a given position
  ///
  __device__ const glm::vec3& getPixel(const glm::ivec2& pos) const;

  /// Get the value of a pixel at a given normalized position
  ///
  __device__ const glm::vec3& getPixel(const glm::vec2& pos) const;

  /// Set the value of a pixel at coordinates x, y
  ///
  __device__ void setPixel(int x, int y, const glm::vec3& value);

  /// Set the value of a pixel at a given position
  ///
  __device__ void setPixel(const glm::ivec2& pos, const glm::vec3& value);

  /// Set the value of a pixel at a given normalized position
  ///
  __device__ void setPixel(const glm::vec2& pos, const glm::vec3& value);

  /// Write the image data to a png file
  ///
  __host__ void writeToFile(const char* filename) const;

  /// Get uv coordinates of a given pixel position
  ///
  __device__ glm::vec2 getUVCoord(const glm::ivec2& pos) const;

  __device__ glm::vec2 getUVCoord(const glm::ivec2& pos,
                                  const glm::vec2& rand) const;

public:
  size_t width;
  size_t height;
  glm::vec3* data;
};

} // namespace cudaray

#endif
