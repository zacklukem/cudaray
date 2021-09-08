#ifndef SRC_MATERIAL
#define SRC_MATERIAL

#include "hittable.h"
#include "texture.h"
#include <curand.h>
#include <curand_kernel.h>

namespace cudaray {

class Hit;

class Material {
public:
  virtual __device__ glm::vec3 emitted(const glm::vec3& p) const;

  virtual __device__ bool scatter(const Ray& r_in, const Hit& rec,
                                  glm::vec3& attenuation, Ray& scattered,
                                  curandState_t& state) const = 0;
};

class Light : public Material {
public:
  __device__ Light();
  virtual __device__ glm::vec3 emitted(const glm::vec3& p) const override;

  virtual __device__ bool scatter(const Ray& r_in, const Hit& rec,
                                  glm::vec3& attenuation, Ray& scattered,
                                  curandState_t& state) const override;
};

class Lambertian : public Material {
public:
  __device__ Lambertian(Texture* texture);
  virtual __device__ bool scatter(const Ray& r_in, const Hit& rec,
                                  glm::vec3& attenuation, Ray& scattered,
                                  curandState_t& state) const override;

public:
  Texture* texture;
};

class Metal : public Material {
public:
  __device__ Metal(const glm::vec3& a, float roughness);

  virtual __device__ bool scatter(const Ray& r_in, const Hit& rec,
                                  glm::vec3& attenuation, Ray& scattered,
                                  curandState_t& state) const override;

public:
  glm::vec3 albedo;
  float roughness;
};
} // namespace cudaray

#endif // SRC_MATERIAL
