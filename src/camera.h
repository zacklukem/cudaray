#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

namespace cudaray {

class Camera {
public:
  /// Create a camera with the given aspect ratio, focal length and origin.
  ///
  __device__ __host__ Camera(float aspect, float focal_length, glm::vec3 origin);

  /// Cast a ray from the camera origin at the given uv coordinate
  ///
  __device__ __host__ Ray getRay(const glm::vec2 &uv) const;

private:
  glm::vec3 origin;
  glm::vec3 lower_left;
  glm::vec3 horizontal;
  glm::vec3 vertical;
};

} // namespace cudaray

#endif
