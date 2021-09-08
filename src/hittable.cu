#include "glm.inl"
#include "hittable.h"
#include "util.h"
#include <cmath>
#include <glm/gtx/intersect.hpp>
using namespace cudaray;

__device__ Sphere::Sphere(glm::vec3 center, float radius, Material* mat)
    : center(center), radius(radius), mat(mat){};

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
  // hit.normal = (hit.point - center) / radius;
  hit.mat = mat;

  glm::vec3 outward_normal = (hit.point - center) / radius;
  hit.setFaceNormal(ray, outward_normal);
  hit.uv = getUV(outward_normal);

  return true;
}

__device__ glm::vec2 Sphere::getUV(const glm::vec3& p) {
  auto theta = glm::acos(-p.y);
  auto phi = atan2(-p.z, p.x) + pi;

  return glm::vec2(phi / (2 * pi), theta / pi);
}

__device__ Mesh::Mesh(glm::vec3* verts, glm::ivec3* tris, size_t num_verts,
                      size_t num_tris, Material* mat)
    : verts(verts), tris(tris), num_tris(num_tris), num_verts(num_verts),
      mat(mat) {
  auto min = glm::vec3(infinity);
  auto max = glm::vec3(-infinity);
  for (int i = 0; i < num_verts; i++) {
    auto& vert = verts[i];
    for (int i = 0; i < 3; i++) {
      if (vert[i] < min[i])
        min[i] = vert[i];
      if (vert[i] > max[i])
        max[i] = vert[i];
    }
  }

  aabb = Aabb(min, max);
}

__device__ bool hitTriangle(const Ray& ray, float t_min, float t_max, Hit& hit,
                            const glm::vec3& v1, const glm::vec3& v2,
                            const glm::vec3& v3) {
  auto bary_pos = glm::vec2();
  float t;
  auto did_intersect = glm::intersectRayTriangle(ray.origin, ray.direction, v1,
                                                 v2, v3, bary_pos, t);

  if (did_intersect) {

    if (t < t_min || t_max < t)
      return false;

    hit.t = t;
    hit.point = ray.at(hit.t);

    hit.setFaceNormal(ray, glm::cross(v1 - v2, v1 - v3));
  }

  return did_intersect;
}

__device__ bool Mesh::hit(const Ray& ray, float t_min, float t_max,
                          Hit& hit) const {

  if (!aabb.hit(ray, t_min, t_max)) {
    return false;
  }

  Hit temp_rec;
  bool hit_anything = false;
  auto closest_so_far = t_max;

  for (int i = 0; i < num_tris; i++) {
    const auto& tri = tris[i];
    const auto& v1 = verts[tri.x];
    const auto& v2 = verts[tri.y];
    const auto& v3 = verts[tri.z];
    bool has_hit =
        hitTriangle(ray, t_min, closest_so_far, temp_rec, v1, v2, v3);
    if (has_hit) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      hit = temp_rec;
    }
  }

  hit.mat = mat;
  return hit_anything;
};

__device__ HittableList::HittableList(){};

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