#include "util.h"

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k) {
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k & (0xffffffff - 1);
}

__device__ float random_float(curandState_t& rand_state) {
  return curand(&rand_state) / (RAND_MAX + 1.0);
}

__device__ float random_float(curandState_t& rand_state, float min, float max) {
  return min + (max - min) * random_float(rand_state);
}

__device__ glm::vec3 random_in_unit_sphere(curandState_t& rand_state) {
  while (true) {
    auto p = glm::vec3(random_float(rand_state, -1, 1),
                       random_float(rand_state, -1, 1),
                       random_float(rand_state, -1, 1));
    if (glm::dot(p, p) >= 1)
      continue;
    return p;
  }
}

__device__ glm::vec3 random_unit_vector(curandState_t& rand_state) {
  return glm::normalize(random_in_unit_sphere(rand_state));
}

__device__ glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n) {
  return v - 2 * glm::dot(v, n) * n;
}