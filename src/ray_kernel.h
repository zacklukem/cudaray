#ifndef RAY_KERNEL_H
#define RAY_KERNEL_H

__global__ void computeRays(const cudaray::Camera* camera,
                            const cudaray::Hittable* world, int width,
                            int height, glm::vec3* image);

#endif
