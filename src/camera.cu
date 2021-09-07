#include "camera.h"

using namespace cudaray;

__device__ __host__ Camera::Camera(float aspect, float focal_length, glm::vec3 origin) {
  auto viewport_height = 2.0f;
  auto viewport_width = aspect * viewport_height;

  this->origin = origin;
  horizontal = glm::vec3(viewport_width, 0.0, 0.0);
  vertical = glm::vec3(0.0, viewport_height, 0.0);
  lower_left = origin - 0.5f * horizontal - 0.5f * vertical -
               glm::vec3(0, 0, focal_length);
}

__device__ __host__ Ray Camera::getRay(const glm::vec2 &uv) const {
  return Ray(origin, lower_left + uv.x * horizontal + uv.y * vertical - origin);
}
