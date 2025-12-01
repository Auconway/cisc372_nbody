//Ayman Tayeb and Austin Conway
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

__global__ void nbody_kernel(vector3* d_pos, vector3* d_vel, double* d_mass, int numEntities) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numEntities) return;

    double pi_x = d_pos[i][0];
    double pi_y = d_pos[i][1];
    double pi_z = d_pos[i][2];

    double vi_x = d_vel[i][0];
    double vi_y = d_vel[i][1];
    double vi_z = d_vel[i][2];

    double accel_sum_x = 0.0;
    double accel_sum_y = 0.0;
    double accel_sum_z = 0.0;

    for (int j = 0; j < numEntities; j++) {
        if (i == j) continue;

        double dx = pi_x - d_pos[j][0];
        double dy = pi_y - d_pos[j][1];
        double dz = pi_z - d_pos[j][2];

        double magnitude_sq = dx*dx + dy*dy + dz*dz;
        double magnitude = sqrt(magnitude_sq);
        
        if (magnitude_sq > 1e-10) {
            double accelmag = -1.0 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
            
            accel_sum_x += accelmag * dx / magnitude;
            accel_sum_y += accelmag * dy / magnitude;
            accel_sum_z += accelmag * dz / magnitude;
        }
    }

    vi_x += accel_sum_x * INTERVAL;
    vi_y += accel_sum_y * INTERVAL;
    vi_z += accel_sum_z * INTERVAL;

    pi_x += vi_x * INTERVAL;
    pi_y += vi_y * INTERVAL;
    pi_z += vi_z * INTERVAL;

    d_vel[i][0] = vi_x;
    d_vel[i][1] = vi_y;
    d_vel[i][2] = vi_z;

    d_pos[i][0] = pi_x;
    d_pos[i][1] = pi_y;
    d_pos[i][2] = pi_z;
}

void compute() {
    vector3 *dev_pos, *dev_vel;
    double *dev_mass;
    size_t size_vec = NUMENTITIES * sizeof(vector3);
    size_t size_mass = NUMENTITIES * sizeof(double);
    cudaError_t err;

    // 1. Allocate memory on the GPU
    err = cudaMalloc((void**)&dev_pos, size_vec);
    if(err != cudaSuccess) { fprintf(stderr, "Cuda Malloc Error (Pos): %s\n", cudaGetErrorString(err)); exit(1); }
    
    err = cudaMalloc((void**)&dev_vel, size_vec);
    if(err != cudaSuccess) { fprintf(stderr, "Cuda Malloc Error (Vel): %s\n", cudaGetErrorString(err)); exit(1); }
    
    err = cudaMalloc((void**)&dev_mass, size_mass);
    if(err != cudaSuccess) { fprintf(stderr, "Cuda Malloc Error (Mass): %s\n", cudaGetErrorString(err)); exit(1); }

    cudaMemcpy(dev_pos, hPos, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vel, hVel, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mass, mass, size_mass, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUMENTITIES + threadsPerBlock - 1) / threadsPerBlock;

    nbody_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_pos, dev_vel, dev_mass, NUMENTITIES);
    
    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dev_pos, size_vec, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dev_vel, size_vec, cudaMemcpyDeviceToHost);

    cudaFree(dev_pos);
    cudaFree(dev_vel);
    cudaFree(dev_mass);
}