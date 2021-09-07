#include "ray.h"

using namespace cudaray;

__host__ __device__ Ray::Ray(glm::vec3 origin, glm::vec3 direction)
    : origin(origin), direction(direction){};

__host__ __device__ glm::vec3 Ray::at(float t) const { return origin + t * direction; }
