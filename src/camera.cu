#include "camera.h"
#include "util.h"

using namespace cudaray;

__device__ __host__ Camera::Camera(float vfov, float aspect, glm::vec3 origin,
                                   glm::vec3 lookat, glm::vec3 up)
    : origin(origin), lookat(lookat), up(up) {
  auto theta = vfov;
  auto h = glm::tan(theta / 2.0f);
  auto viewport_height = 2.0f * h;
  auto viewport_width = aspect * viewport_height;

  auto focal_length = 1.0f;

  auto w = glm::normalize(origin - lookat);
  auto u = glm::normalize(glm::cross(up, w));
  auto v = glm::cross(w, u);

  horizontal = viewport_width * u;
  vertical = viewport_height * v;
  lower_left = origin - horizontal / 2.0f - vertical / 2.0f - w;

  // horizontal = glm::vec3(viewport_width, 0.0, 0.0);
  // vertical = glm::vec3(0.0, viewport_height, 0.0);
  // lower_left = origin - 0.5f * horizontal - 0.5f * vertical -
  //              glm::vec3(0, 0, focal_length);
}

__device__ __host__ Ray Camera::getRay(const glm::vec2& uv) const {
  return Ray(origin, lower_left + uv.x * horizontal + uv.y * vertical - origin);
}
