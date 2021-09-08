#include "texture.h"

using namespace cudaray;

__device__ SolidTexture::SolidTexture(const glm::vec3& color) : color(color){};

__device__ glm::vec3 SolidTexture::value(const glm::vec2& uv,
                                         const glm::vec3& point) const {
  return color;
}

__device__ CheckerTexture::CheckerTexture(Texture* even, Texture* odd)
    : even(even), odd(odd){};
__device__ CheckerTexture::CheckerTexture(const glm::vec3& even,
                                          const glm::vec3& odd)
    : even(new SolidTexture(even)), odd(new SolidTexture(odd)){};

__device__ glm::vec3 CheckerTexture::value(const glm::vec2& uv,
                                           const glm::vec3& point) const {
  auto sines = sin(10 * point.x) * sin(10 * point.y) * sin(10 * point.z);

  if (sines < 0)
    return odd->value(uv, point);
  else
    return even->value(uv, point);
}