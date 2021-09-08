#include "camera.h"
#include "hittable.h"
#include "image.h"
#include "ray.h"
#include "util.h"

#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>

__device__ glm::vec3 solarAttn(const glm::vec3& vec) {
  auto len = glm::length(vec);
  // return glm::max(len * 100.0f - 99.0f, 0.0f);
  return len < 0.3f ? glm::vec3(3.0f) : glm::vec3(0.5f, 0.7f, 1.0f);
}

__device__ glm::vec3 rayColor(curandState_t& rand_state,
                              const cudaray::Ray& ray,
                              const cudaray::Hittable& world, int depth) {
  glm::vec3 color(0.0f, 0.0f, 0.0f);
  cudaray::Ray current_ray = cudaray::Ray(ray.origin, ray.direction);
  glm::vec3 factor = glm::vec3(1.0f);
  int i = 0;
  while (i < depth) {
    i++;
    cudaray::Hit rec;
    if (world.hit(current_ray, 0.001, infinity, rec)) {
      // Ray scattered;
      glm::vec3 attenuation;
      if (rec.mat->scatter(current_ray, rec, attenuation, current_ray,
                           rand_state)) {
        factor *= attenuation;
      } else {
        color = rec.mat->emitted(glm::vec3());
        break;
      }
    } else {
// #define SUN_ENABLED
#ifdef SUN_ENABLED
      glm::vec3 unit_direction = glm::normalize(current_ray.direction);

      glm::vec3 cross_d_sun =
          glm::cross(glm::normalize(glm::vec3(-1)), unit_direction);
      // if (almostZero(cross_d_sun)) {
      color = solarAttn(cross_d_sun);
#else
      color = glm::vec3(0.03f);
#endif
      // } else {
      // auto t = 0.5f * (unit_direction.y + 1.0f);
      // color = 0.3f *
      // ((1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0));
      // }
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
