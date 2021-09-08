#include "camera.h"
#include "cuda.h"
#include "glm.inl"
#include "hittable.h"
#include "image.h"
#include "obj.h"
#include "ray_kernel.h"
#include "util.h"
#include <iostream>

__global__ void populateWorld(cudaray::HittableList* world, glm::vec3* verts,
                              glm::ivec3* tris, size_t num_tris,
                              size_t num_vert) {

  for (int i = 0; i < num_vert; i++) {
    verts[i] *= 0.1f;
    verts[i].z -= 1.0f;
    // verts[i].y += 1.0f;
  }

  // auto mesh =
  //     new cudaray::Mesh(verts, tris, num_vert, num_tris,
  //                       new cudaray::Lambertian(glm::vec3(0.5f, 0.5f,
  //                       0.8f)));

  // world->add(mesh);
  world->add(new cudaray::Sphere(glm::vec3(0.0f, 1.0f, -1.0f), 0.2f,
                                 new cudaray::Light()));
  world->add(
      new cudaray::Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.4f,
                          new cudaray::Lambertian(new cudaray::SolidTexture(
                              glm::vec3(0.8f, 0.5f, 0.5f)))));
  world->add(new cudaray::Sphere(
      glm::vec3(1.0f, 0.0f, -1.0f), 0.2f,
      new cudaray::Metal(glm::vec3(0.7f, 0.3f, 0.7f), 0.5f)));
  world->add(new cudaray::Sphere(
      glm::vec3(-1.0f, 0.0f, -1.0f), 0.2f,
      new cudaray::Metal(glm::vec3(0.7f, 0.7f, 0.7f), 0.0f)));
  world->add(new cudaray::Sphere(
      glm::vec3(0.0f, -100.5, -1.0f), 100.0f,
      new cudaray::Lambertian(new cudaray::CheckerTexture(
          glm::vec3(0.5f, 0.8f, 0.5f), glm::vec3(0.5f, 0.5f, 0.8f)))));
  // world->add(new cudaray::Sphere(
  //     glm::vec3(0.0f, -100.5, -1.0f), 100.0f,
  //     new cudaray::Light()));
}

const int width = 600;
const int height = 600;
const int numSamples = 1000;

static_assert(width % 15 == 0, "width must be multiple of 15");
static_assert(height % 15 == 0, "height must be multiple of 15");

int main() {
  cudaSetDevice(1);

  auto teapot = cudaray::loadObj("tea.obj");

  glm::vec3* verts;
  glm::ivec3* tris;
  cudaMalloc(&verts, sizeof(glm::vec3) * teapot.vertices.size());
  cudaMalloc(&tris, sizeof(glm::ivec3) * teapot.faces.size());
  cudaMemcpy(verts, &teapot.vertices.front(),
             sizeof(glm::vec3) * teapot.vertices.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(tris, &teapot.faces.front(),
             sizeof(glm::ivec3) * teapot.faces.size(), cudaMemcpyHostToDevice);

  auto d_sphere = cudaray::make_in_device<cudaray::HittableList>();

  populateWorld<<<1, 1>>>(d_sphere, verts, tris, teapot.faces.size(),
                          teapot.vertices.size());

  auto d_camera = cudaray::make_in_device<cudaray::Camera>(
      pi / 10.0f, (float)(width / height), glm::vec3(5.0f),
      glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));

  glm::vec3* d_image;
  cudaMalloc(&d_image, sizeof(glm::vec3) * width * height);

  dim3 numBlocks(width / 15, height / 15, numSamples);
  dim3 threadsPerBlock(15, 15);
  computeRays<<<numBlocks, threadsPerBlock>>>(d_camera, d_sphere, width, height,
                                              d_image);

  cudaray::free_in_device<cudaray::HittableList>(d_sphere);
  // cudaray::free_in_device<cudaray::Image>(d_image);
  cudaray::free_in_device<cudaray::Camera>(d_camera);

  glm::vec3* image_data =
      (glm::vec3*)malloc(sizeof(glm::vec3) * width * height);
  cudaMemcpy(image_data, d_image, sizeof(glm::vec3) * width * height,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; i++) {
    auto scale = 1.0 / numSamples;
    image_data[i].r = sqrt(scale * image_data[i].r);
    image_data[i].g = sqrt(scale * image_data[i].g);
    image_data[i].b = sqrt(scale * image_data[i].b);
  };

  cudaray::Image image(image_data, width, height);
  image.writeToFile("test.png");

  cudaFree(d_image);

  auto err = cudaGetLastError();
  std::cout << cudaGetErrorString(err) << "\n";

  // image.writeToFile("test.png");
}
