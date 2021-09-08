#ifndef SRC_TEXTURE
#define SRC_TEXTURE

#include "glm.inl"

namespace cudaray {

class Texture {
public:
  virtual __device__ glm::vec3 value(const glm::vec2& uv,
                                     const glm::vec3& point) const = 0;
};

class SolidTexture : public Texture {
public:
  __device__ SolidTexture(const glm::vec3& color);
  virtual __device__ glm::vec3 value(const glm::vec2& uv,
                                     const glm::vec3& point) const override;

private:
  glm::vec3 color;
};

class CheckerTexture : public Texture {
public:
  __device__ CheckerTexture(Texture* even, Texture* odd);
  __device__ CheckerTexture(const glm::vec3& even, const glm::vec3& odd);
  virtual __device__ glm::vec3 value(const glm::vec2& uv,
                                     const glm::vec3& point) const override;

private:
  Texture* even;
  Texture* odd;
};

} // namespace cudaray

#endif // SRC_TEXTURE
