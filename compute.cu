//Ayman Tayeb and Austin Conway

#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include "config.h"


/*
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}

*/

__global__ void nbody_kernel(vector3* d_pos, vector3* d_vel, double* d_mass, int numEntities) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numEntities) {
        return;
    }

    double pos_x = d_pos[i][0];
    double pos_y = d_pos[i][1];
    double pos_z = d_pos[i][2];

    double vel_x = d_vel[i][0];
    double vel_y = d_vel[i][1];
    double vel_z = d_vel[i][2];

    double accel_x_sum = 0.0;
    double accel_y_sum = 0.0;
    double accel_z_sum = 0.0;

    for (int j = 0; j < numEntities; j++) {
        if (i == j) {
            continue;
        }

        double delta_x = pos_x - d_pos[j][0];
        double delta_y = pos_y - d_pos[j][1];
        double delta_z = pos_z - d_pos[j][2];

        double distance_squared = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
        double distance = sqrt(distance_squared);
        
        if (distance_squared > 1e-10) {
            double acceleration_magnitude = -1.0 * GRAV_CONSTANT * d_mass[j] / distance_squared;
            
            accel_x_sum += acceleration_magnitude * delta_x / distance;
            accel_y_sum += acceleration_magnitude * delta_y / distance;
            accel_z_sum += acceleration_magnitude * delta_z / distance;
        }
    }

    vel_x += accel_x_sum * INTERVAL;
    vel_y += accel_y_sum * INTERVAL;
    vel_z += accel_z_sum * INTERVAL;

    pos_x += vel_x * INTERVAL;
    pos_y += vel_y * INTERVAL;
    pos_z += vel_z * INTERVAL;

    d_vel[i][0] = vel_x;
    d_vel[i][1] = vel_y;
    d_vel[i][2] = vel_z;

    d_pos[i][0] = pos_x;
    d_pos[i][1] = pos_y;
    d_pos[i][2] = pos_z;
}

void compute() {
    vector3 *dev_pos, *dev_vel;
    double *dev_mass;
    double *device_mass;
    size_t vector_size = NUMENTITIES * sizeof(vector3);
    size_t mass_size = NUMENTITIES * sizeof(double);
    size_t size_vec = NUMENTITIES * sizeof(vector3);
    size_t size_mass = NUMENTITIES * sizeof(double);
    cudaError_t err;

    // this will allocate memory on the GPU
    err = cudaMalloc((void**)&dev_pos, size_vec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to allocate GPU memory for positions (dev_pos): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&dev_vel, size_vec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to allocate GPU memory for velocities (dev_vel): %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&dev_mass, size_mass);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Unable to allocate GPU memory for masses (dev_mass): %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemcpy(dev_pos, hPos, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vel, hVel, size_vec, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mass, mass, size_mass, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (NUMENTITIES + threads_per_block - 1) / threads_per_block;

    nbody_kernel<<<blocks_per_grid, threads_per_block>>>(dev_pos, dev_vel, dev_mass, NUMENTITIES);
    
    // we are checking for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, dev_pos, size_vec, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dev_vel, size_vec, cudaMemcpyDeviceToHost);

    cudaFree(dev_pos);
    cudaFree(dev_vel);
    cudaFree(dev_mass);
}