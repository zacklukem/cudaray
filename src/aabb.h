#ifndef SRC_AABB
#define SRC_AABB

#include "ray.h"

namespace cudaray {

class Aabb {
public:
  __device__ Aabb() {}
  __device__ Aabb(const glm::vec3& a, const glm::vec3& b) {
    minimum = a;
    maximum = b;
  }

  __device__ glm::vec3 min() const { return minimum; }
  __device__ glm::vec3 max() const { return maximum; }

  __device__ bool hit(const Ray& r, float t_min, float t_max) const {
    for (int a = 0; a < 3; a++) {
      float t0 = glm::min((minimum[a] - r.origin[a]) / r.direction[a],
                          (maximum[a] - r.origin[a]) / r.direction[a]);
      float t1 = glm::max((minimum[a] - r.origin[a]) / r.direction[a],
                          (maximum[a] - r.origin[a]) / r.direction[a]);
      t_min = glm::max(t0, t_min);
      t_max = glm::min(t1, t_max);
      if (t_max <= t_min)
        return false;
    }
    return true;
  }

  glm::vec3 minimum;
  glm::vec3 maximum;
};
} // namespace cudaray

#endif // SRC_AABB
