#ifndef HITTABLE_H
#define HITTABLE_H

#include "glm.inl"
#include "cuda.h"
#include "ray.h"
#include "vector.h"

namespace cudaray {

struct Hit {
  glm::vec3 point;
  glm::vec3 normal;
  float t;
  bool front_face;

  __device__ inline void setFaceNormal(const Ray& ray, const glm::vec3& outward_normal) {
    front_face = glm::dot(ray.direction, outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Hittable {
public:
  virtual __device__ bool hit(const Ray& ray, float t_min, float t_max,
                              Hit& hit) const = 0;
};

class Sphere : public Hittable {
public:
  __device__ Sphere(glm::vec3 center, float radius);
  __device__ bool hit(const Ray& ray, float t_min, float t_max,
                      Hit& hit) const override;

public:
  glm::vec3 center;
  float radius;
};

class HittableList : public Hittable {
public:
  __device__ HittableList();

  __device__ void add(Hittable* object);

  __device__ bool hit(const Ray& ray, float t_min, float t_max,
                      Hit& hit) const override;

private:
  csl::vector<Hittable*> objects;
  size_t len;
};

} // namespace cudaray

#endif
