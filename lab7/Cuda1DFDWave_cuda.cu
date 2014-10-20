#include "Cuda1DFDWave_cuda.cuh"



/* TODO: You'll need a kernel, as well as any helper functions
to call it */

/* kernel to calculate new displacements */
__global__ void GenerateDisplacements(float* dev_Data, int oldStart, 
                                    int currentStart, int newStart,
                                    float courantSquared, int numberOfNodes) {

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    //printf("INSIDE KERNEL? \n");

    //dev_Data[currentStart + index] = 5.0;
    //dev_Data[oldStart + index] = 5.0;
    //dev_Data[newStart+index] = 5.0;
    
    
    // ignore both left and right boundary conditions (already set on GPU) 
    while (index < numberOfNodes - 1) {      
        if (index > 0) {
            // compute next displacement:
            // y(x,t+1) = 2 * y(x,t) - y(x,t-1) + courantSquared(y(x+1,t)
            //      - 2 * y(x,t) + y(x-1,t))
            
            
            dev_Data[newStart + index] = 2 * dev_Data[currentStart + index] 
                - dev_Data[oldStart + index] + courantSquared * 
                (dev_Data[currentStart + index + 1] - 
                2 * dev_Data[currentStart + index] + 
                dev_Data[currentStart + index - 1]); 
            //dev_Data[newStart + index] = 5.0;
        }

        index += blockDim.x * gridDim.x; // increment index to next value
    }
}

/* helper function to call the kernel */
void kernelCall(float* dev_Data, int oldStart, int currentStart, int newStart,
                                 float courantSquared, int numberOfNodes, 
                                  unsigned int blocks, unsigned int threadsPerBlock) {

    GenerateDisplacements <<<blocks, threadsPerBlock>>> (dev_Data, oldStart, 
                            currentStart, newStart, courantSquared, numberOfNodes);
}