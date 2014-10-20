/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH


/* If you have any functions in your .cu file that need to be
accessed from the outside, declare them here */

__global__ void GenerateDisplacements(float* dev_Data, int oldStart, int currentStart, int newStart,
                                 float courantSquared, int numberOfNodes);

/* helper function to call the kernel */
void kernelCall(float* dev_Data, int oldStart, int currentStart, int newStart,
                                 float courantSquared, int numberOfNodes,
				unsigned int blocks, unsigned int threadsPerBlock);

#endif // CUDA_1D_FD_WAVE_CUDA_CUH
