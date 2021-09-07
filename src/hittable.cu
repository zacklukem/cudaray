#include "glm.inl"
#include "hittable.h"
using namespace cudaray;

__device__ Sphere::Sphere(glm::vec3 center, float radius)
    : center(center), radius(radius){};

__device__ bool Sphere::hit(const Ray& ray, float t_min, float t_max,
                            Hit& hit) const {
  glm::vec3 oc = ray.origin - center;
  auto a = glm::dot(ray.direction, ray.direction);
  auto half_b = glm::dot(oc, ray.direction);
  auto c = glm::dot(oc, oc) - radius * radius;

  auto discriminant = half_b * half_b - a * c;
  if (discriminant < 0)
    return false;
  auto sqrtd = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrtd) / a;
    if (root < t_min || t_max < root)
      return false;
  }

  hit.t = root;
  hit.point = ray.at(hit.t);
  hit.normal = (hit.point - center) / radius;

  glm::vec3 outward_normal = (hit.point - center) / radius;
  hit.setFaceNormal(ray, outward_normal);

  return true;
}

__device__ HittableList::HittableList(){ };

__device__ void HittableList::add(Hittable* object) {
  objects.push_back(object);
}

__device__ bool HittableList::hit(const Ray& ray, float t_min, float t_max,
                                  Hit& hit) const {
  Hit temp_rec;
  bool hit_anything = false;
  auto closest_so_far = t_max;

  for (size_t i = 0; i < objects.size(); i++) {
    auto& object = objects[i];
    if (object->hit(ray, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      hit = temp_rec;
    }
  }

  return hit_anything;
}