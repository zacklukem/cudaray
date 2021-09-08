#include "material.h"

using namespace cudaray;

__device__ glm::vec3 Material::emitted(const glm::vec3& p) const {
  return glm::vec3(0.0f);
};

__device__ Light::Light(){};

__device__ glm::vec3 Light::emitted(const glm::vec3& p) const {
  return glm::vec3(5.0f);
};

__device__ bool Light::scatter(const Ray& r_in, const Hit& rec,
                               glm::vec3& attenuation, Ray& scattered,
                               curandState_t& state) const {
  return false;
};

__device__ Lambertian::Lambertian(Texture* texture) : texture(texture) {}

__device__ bool Lambertian::scatter(const Ray& r_in, const Hit& rec,
                                    glm::vec3& attenuation, Ray& scattered,
                                    curandState_t& state) const {
  auto scatter_direction = rec.normal + random_unit_vector(state);
  scattered = Ray(rec.point, scatter_direction);
  attenuation = texture->value(rec.uv, rec.point);
  return true;
}

__device__ Metal::Metal(const glm::vec3& a, float roughness)
    : albedo(a), roughness(roughness) {}

__device__ bool Metal::scatter(const Ray& r_in, const Hit& rec,
                               glm::vec3& attenuation, Ray& scattered,
                               curandState_t& state) const {
  glm::vec3 reflected = reflect(r_in.direction, rec.normal);
  scattered = Ray(rec.point, reflected + roughness * random_unit_vector(state));
  attenuation = albedo;
  return (glm::dot(scattered.direction, rec.normal) > 0);
}