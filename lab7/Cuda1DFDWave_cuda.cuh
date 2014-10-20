/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 
 * 
 * To be completed by students as Assignment 7 of CS 179
 */


#include <cstdio>
#include <cuda_runtime.h>
 

#define gpuErrchk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH



/* TODO: Put things in your header here */


__global__ void GenerateDisplacements(float* dev_Data, int oldStart, int currentStart, int newStart,
                                 float courantSquared, int numberOfNodes);

/* helper function to call the kernel */
void kernelCall(float* dev_Data, int oldStart, int currentStart, int newStart,
                                 float courantSquared, int numberOfNodes,
				unsigned int blocks, unsigned int threadsPerBlock);




#endif // CUDA_1D_FD_WAVE_CUDA_CUH
