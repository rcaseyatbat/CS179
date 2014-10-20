// -*- C++ -*-
#include <cstdio>

#include <cuda_runtime.h>
#include <cuda.h>

#include "Main1_cuda.cuh"
#include "cuPrintf.cu"

//since we can't really dynamically size this array,
//let's leave its size at the default polynomial order
__constant__ float constant_c[10];


__global__
void
cudaSum_atomic_kernel(const float* const inputs,
                                     unsigned int numberOfInputs,
                                     const float* const c,
                                     unsigned int polynomialOrder,
                                     float* output) {
    
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float partial_sum = 0.0;
    
    while (index < numberOfInputs) {
        // calculate polynomial value at inputs[index]
        float r = inputs[index];
        float result = 0.0;
        float power = 1.0;

        for (unsigned int i = 0; i < polynomialOrder; i++) {
            result += (c[i] * power);
            power *= r;
        }

        partial_sum += result; // add result of P(r) to partial_sum

        // increment index to next value, if this thread has to handle
        // multiple elements
        index += blockDim.x * gridDim.x;
    }
    atomicAdd(output, partial_sum);
    

}

__global__
void
cudaSum_linear_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs, 
                                  const float* const c,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    
    
    extern __shared__ float partial_outputs[];

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float partial_sum = 0.0;
    
    while (index < numberOfInputs) {

        // calculate polynomial value at inputs[index]
        float r = inputs[index];
        float result = 0.0;
        float power = 1.0;

        for (unsigned int i = 0; i < polynomialOrder; i++) {
            result += (c[i] * power);
            power *= r;
        }

        partial_sum += result; // add result of P(r) to partial_sum

        index += blockDim.x * gridDim.x; // increment index to next value
    }
    partial_outputs[threadIdx.x] = partial_sum;

    // Make all threads in the block finish before computing
    syncthreads(); 

    // Here, start with the first thread's partial sum.  Add the rest of the threads'
    // partial sums, so that partial_sum contains the sum from all of threads
    // of the block. 
    if (threadIdx.x == 0) {
        for (unsigned int threadIndex = 1; threadIndex < blockDim.x; threadIndex++) {
            partial_sum += partial_outputs[threadIndex];
        }

        // Now, finally accumulate (we've already added together all the partial sums)
        // Note that atomicAdd is called is once per block now, instead of per thread
        atomicAdd(output, partial_sum);
    }    
}
 

/* Used in Assignment 2. Coming soon! */
__global__
void
cudaSum_divtree_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs, 
                                  const float* const c,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    
    extern __shared__ float partial_outputs[];

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    float partial_sum = 0.0;
    
    while (index < numberOfInputs) {

        // calculate polynomial value at inputs[index]
        float r = inputs[index];
        float result = 0.0;
        float power = 1.0;

        for (unsigned int i = 0; i < polynomialOrder; i++) {
            result += (c[i] * power);
            power *= r;
        }

        partial_sum += result; // add result of P(r) to partial_sum

        index += blockDim.x * gridDim.x; // increment index to next value
    }
    partial_outputs[threadIdx.x] = partial_sum;


    // Make all threads in the block finish before computing
    syncthreads(); 

    

    cuPrintf("p[%d] = %f \n", threadIdx.x, partial_outputs[threadIdx.x]);

    
    int offset = 1;
    cuPrintf("%d \n", offset < blockDim.x);
    
    while (offset < blockDim.x) {

        if (threadIdx.x == 0) {
            cuPrintf("OFFSET: %d \n", offset);
            cuPrintf("MOD: %d \n", threadIdx.x % offset == 0);
        }
        
        if (threadIdx.x % offset == 0) {
            float add = partial_outputs[(int)threadIdx.x + offset];
            float current = partial_outputs[(int)threadIdx.x];
            //cuPrintf("add: %f   current: %f \n", add, current);
            cuPrintf("ADD: %d \n", (int)threadIdx.x);
            //partial_outputs[(int)threadIdx.x] = 1.0;//add + current;//add + current;
            //partial_outputs[threadIdx.x] = add;
            partial_outputs[threadIdx.x] = partial_outputs[threadIdx.x] + partial_outputs[threadIdx.x + offset];
        }
        
        offset = offset * 2;
        syncthreads();
    }
    

    //cuPrintf("PP[%d] = %f \n", threadIdx.x, partial_outputs[threadIdx.x]);

    /*

    cuPrintf("p[0] = %d \n", partial_outputs[0]);

    if (threadIdx.x == 0) {
        atomicAdd(output, partial_outputs[0]);
    }
    */
    /*
    // Here, start with the first thread's partial sum.  Add the rest of the threads'
    // partial sums, so that partial_sum contains the sum from all of threads
    // of the block. 
    if (threadIdx.x == 0) {
        for (unsigned int threadIndex = 1; threadIndex < blockDim.x; threadIndex++) {
            partial_sum += partial_outputs[threadIndex];
        }

        // Now, finally accumulate (we've already added together all the partial sums)
        // Note that atomicAdd is called is once per block now, instead of per thread
        atomicAdd(output, partial_sum);
    }   
    */ 
}

/* Used in Assignment 2. Coming soon! */
__global__
void
cudaSum_nondivtree_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs, 
                                  const float* const c,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    

}

/* Used in Assignment 2. Coming soon! */
__global__
void
cudaSum_constmem_kernel(const float* const inputs, 
                                  unsigned int numberOfInputs,
                                  unsigned int polynomialOrder, 
                                  float * output) {
    

}




void
cudaSumPolynomials(const float* const input,
                            const size_t numberOfInputs,
                            const float* const c,
                            const size_t polynomialOrder,
                            const Style style,
                            const unsigned int maxBlocks,
                            float * const output) {


    //Input values (your "r" values) go here on the GPU
    float *dev_input;
    
    //Your polynomial coefficients go here (GPU)
    float *dev_c;
    
    //Your output will go here (GPU)
    float *dev_output;
    
    // Allocate memory on the GPU for our inputs
    cudaMalloc((void **) &dev_input, numberOfInputs*sizeof(float));
    cudaMemcpy(dev_input, input, numberOfInputs*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **) &dev_c, polynomialOrder*sizeof(float));
    cudaMemcpy(dev_c, c, polynomialOrder*sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory on the GPU for outputs
    cudaMalloc((void **) &dev_output, sizeof(float));
    cudaMemset(dev_output, 0, sizeof(float)); // make sure to initialize output to 0!

    cudaPrintfInit();    
    
    const unsigned int threadsPerBlock = 512;
    const unsigned int blocks 
                = min((float)maxBlocks, 
                        ceil(numberOfInputs/(float)threadsPerBlock));

    if (style == mutex) {
        cudaSum_atomic_kernel<<<blocks, threadsPerBlock>>>(
                dev_input, numberOfInputs, dev_c, polynomialOrder, dev_output);
    } else if (style == linear) {
        cudaSum_linear_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                dev_c, polynomialOrder, dev_output);
    } else if (style == divtree) {
        cudaSum_divtree_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                dev_c, polynomialOrder, dev_output);
    } else if (style == nondivtree) {
        cudaSum_nondivtree_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                dev_c, polynomialOrder, dev_output);
    } else if (style == constmem) {
        
        //initialize the constant memory
        cudaMemcpyToSymbol("constant_c", c, polynomialOrder * sizeof(float),
                0, cudaMemcpyHostToDevice);
        
        cudaSum_constmem_kernel<<<blocks, threadsPerBlock, 
                threadsPerBlock*sizeof(float)>>>(dev_input, numberOfInputs, 
                polynomialOrder, dev_output);
    } else {
        printf("Unknown style\n");
    }

    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    // Copy output from device to host
    cudaMemcpy(output, dev_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(dev_input);
    cudaFree(dev_c);
    cudaFree(dev_output);
}
