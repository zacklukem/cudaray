#ifndef SRC_UTIL
#define SRC_UTIL

#include <curand.h>
#include <curand_kernel.h>
#include <glm.inl>

#include <limits>

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k);
__device__ float random_float(curandState_t& rand_state);
__device__ float random_float(curandState_t& rand_state, float min, float max);
__device__ glm::vec3 random_in_unit_sphere(curandState_t& rand_state);
__device__ glm::vec3 random_unit_vector(curandState_t& rand_state);
__device__ glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n);

#endif // SRC_UTIL
