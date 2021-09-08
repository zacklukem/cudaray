#ifndef SRC_HITTABLE
#define SRC_HITTABLE

#include "aabb.h"
#include "cuda.h"
#include "glm.inl"
#include "material.h"
#include "ray.h"
#include "util.h"
#include <csl/vector.h>
#include <curand.h>
#include <curand_kernel.h>

namespace cudaray {

class Material;

struct Hit {
  glm::vec3 point;
  glm::vec3 normal;
  Material* mat;
  float t;
  glm::vec2 uv;
  bool front_face;

  __device__ inline void setFaceNormal(const Ray& ray,
                                       const glm::vec3& outward_normal) {
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
  __device__ Sphere(glm::vec3 center, float radius, Material* mat);
  __device__ bool hit(const Ray& ray, float t_min, float t_max,
                      Hit& hit) const override;

public:
  __device__ static glm::vec2 getUV(const glm::vec3& p);
  glm::vec3 center;
  float radius;
  Material* mat;
};

class Mesh : public Hittable {
public:
  __device__ Mesh(glm::vec3* verts, glm::ivec3* tris, size_t num_verts,
                  size_t num_tris, Material* mat);
  __device__ bool hit(const Ray& ray, float t_min, float t_max,
                      Hit& hit) const override;

public:
  Aabb aabb;
  const glm::vec3* verts;
  // glm::vec3* normals;
  const glm::ivec3* tris;
  size_t num_verts;
  size_t num_tris;
  Material* mat;
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

#endif // SRC_HITTABLE
