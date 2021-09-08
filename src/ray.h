#ifndef RAY_H
#define RAY_H

#include "cuda.h"
#include "glm.inl"

namespace cudaray {

class Ray {
public:
  __host__ __device__ Ray(glm::vec3 origin, glm::vec3 direction);

  __host__ __device__ glm::vec3 at(float t) const;

  glm::vec3 origin;
  glm::vec3 direction;
};

} // namespace cudaray

#endif
