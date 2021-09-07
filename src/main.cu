#include <iostream>
#include "cuda.h"
#include "util.h"
#include "camera.h"
#include "hittable.h"
#include "image.h"
#include "glm.inl"
#include "ray_kernel.h"

__global__ void populateWorld(cudaray::HittableList* world) {
  world->add(new cudaray::Sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f));
  world->add(new cudaray::Sphere(glm::vec3(0.0f, -100.5, -1.0f), 100.0f));
}

int main() {
  const int width = 640;
  const int height = 640;
  const int numSamples = 100;

  auto d_sphere = cudaray::make_in_device<cudaray::HittableList>();

  populateWorld<<<1, 1>>>(d_sphere);

  auto d_camera = cudaray::make_in_device<cudaray::Camera>(1.0f, 1.0f, glm::vec3());

  glm::vec3* d_image;
  cudaMalloc(&d_image, sizeof(glm::vec3) * width * height);

  dim3 numBlocks(20, 20, numSamples);
  dim3 threadsPerBlock(32, 32);
  computeRays<<<numBlocks, threadsPerBlock>>>(d_camera, d_sphere, width, height, d_image);

  cudaray::free_in_device<cudaray::HittableList>(d_sphere);
  // cudaray::free_in_device<cudaray::Image>(d_image);
  cudaray::free_in_device<cudaray::Camera>(d_camera);

  glm::vec3* image_data = (glm::vec3*)malloc(sizeof(glm::vec3)*width*height);
  cudaMemcpy(image_data, d_image, sizeof(glm::vec3)*width*height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width*height; i++) {
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
