#include "camera.h"
#include "hittable.h"
#include "image.h"
#include "ray.h"
#include "util.h"

#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

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

__device__ glm::vec3 rayColor(curandState_t& rand_state,
                              const cudaray::Ray& ray,
                              const cudaray::Hittable& world, int depth) {
  glm::vec3 color(0.0f, 0.0f, 0.0f);
  cudaray::Ray current_ray = cudaray::Ray(ray.origin, ray.direction);
  float factor = 1.0f;
  int i = 0;
  while (i < depth) {
    cudaray::Hit rec;
    if (world.hit(current_ray, 0.001, infinity, rec)) {
      glm::vec3 target =
          rec.point + rec.normal + random_in_unit_sphere(rand_state);

      current_ray.origin = rec.point;
      current_ray.direction = target - rec.point;
      i++;
      factor *= 0.5f;

    } else {
      glm::vec3 unit_direction = glm::normalize(ray.direction);
      auto t = 0.5f * (unit_direction.y + 1.0f);
      color =
          (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
      break;
    }
  }
  return color * factor;
}

__global__ void computeRays(const cudaray::Camera* camera,
                            const cudaray::Hittable* world, int width,
                            int height, glm::vec3* image_data) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  curandState_t rand_state;

  curand_init(hash(x + y + hash(x) + hash(y) + hash(blockIdx.z) + blockIdx.z),
              0, 0, &rand_state);

  cudaray::Image image(image_data, width, height);

  auto pixel = glm::ivec2(x, y);
  auto uv = image.getUVCoord(pixel, glm::vec2(random_float(rand_state) / 2.0f,
                                              random_float(rand_state) / 2.0f));
  uv.y = 1.0f - uv.y; // flip vertical
  auto ray = camera->getRay(uv);

  glm::vec3 out_color = rayColor(rand_state, ray, *world, 100);

  image.setPixel(pixel, image.getPixel(pixel) + out_color);
}
