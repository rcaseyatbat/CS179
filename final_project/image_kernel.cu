/*
 */

#include "helper_math.h"
#include "image_kernel.cuh"
#include <helper_cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <png.h>

typedef unsigned int uint;

#define C_PI 3.141592653589793238462643383279502884197169399375
// macro for error-handling
#define gpuErrchk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

typedef struct {
    float4 m[3];
} float3x4;

// color conversion functions
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ uint rFloatToInt(float r)
{
    r = __saturatef(r);   // clamp to [0.0, 1.0]
    return (uint(r*255)<<24) | (uint(r*255)<<16) | (uint(r*255)<<8) | uint(r*255);
}


void image_proc(GLuint *d_input, uint imageW, uint imageH, GLuint *d_output)
{
    for (uint x = 0; x < imageW; ++x) {
        for (uint y = 0; y < imageH; ++y) {

        uint i = y * imageW + x;
        d_output[i] = ~d_input[i]; // simply invert all the bits!
        }
    }

}

__global__ void d_image_proc(uint *d_input, uint imageW, uint imageH, uint *d_output)
{
    // Calculate indices
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint i = y * imageW + x;

    if ((x >= imageW) || (y >= imageH)) return;

    d_output[i] = ~d_input[i]; // simply invert all the bits in the color!
}

// CPU version
void image_Edge(GLuint *d_input, uint imageW, uint imageH, GLuint *d_output)
{

    for (uint x = 0; x < imageW; ++x) {
        for (uint y = 0; y < imageH; ++y) {

        uint i = y * imageW + x;

        // avoid border cases
        if (x == 0 || x == imageW - 1 || y == 0 || y == imageH -1) {
            // if we're on the edgejust set the pixel to be black
            d_output[i] = 0;
        } else {
            // apply gaussian filter here
            uint up = (y+1) * imageW + x;
            uint down = (y-1) * imageW + x;
            uint left = y * imageW + (x - 1);
            uint right = y * imageW + (x + 1);
            uint upRight = (y+1) * imageW + (x+1);
            uint upLeft = (y+1) * imageW + (x-1);
            uint downRight = (y-1) * imageW + (x+1);
            uint downLeft = (y-1) * imageW + (x-1);

            int indices [9] = {upLeft, up, upRight, left, i, right, downLeft, down, downRight};

            int rollingR = 0;
            int rollingG = 0;
            int rollingB = 0;
            int rollingA = 0;

            for (int j = 0; j < 9; j++) {
                uint colorI = d_input[indices[j]];
                uint aI = colorI >> 24 & 255; 
                uint rI = colorI >> 16 & 255; 
                uint gI = colorI >> 8 & 255; 
                uint bI = colorI >> 0 & 255; 
                if (j == 4) {
                    rollingR += 8 * rI;
                    rollingG += 8 * gI;
                    rollingB += 8 * bI;
                    rollingA += 8 * aI;
                } else {
                    rollingR -= rI;
                    rollingG -= gI;
                    rollingB -= bI;
                    rollingA -= aI;
                }
            }

            // divide by the factor here if we don't add to 0
            //rollingR = rollingR / 9;
            //rollingG = rollingG / 9;
            //rollingB = rollingB / 9;

            rollingR = min(max(int(1 * rollingR + 0), 0), 255); 
            rollingG = min(max(int(1 * rollingG + 0), 0), 255); 
            rollingB = min(max(int(1 * rollingB + 0), 0), 255); 

            uint newcolor = uint(255 << 24 | rollingR << 16 | rollingG << 8 | rollingB);
            d_output[i] = newcolor;
            }
        }
    }
}

// GPU version
__global__ void d_image_Edge(uint *d_input, uint imageW, uint imageH, uint *d_output)
{
    // Calculate indices
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Calculate screen coordinates based on indices
    if ((x >= imageW) || (y >= imageH)) return;

    uint i = y * imageW + x;

    // avoid border cases
    if (x == 0 || x == imageW - 1 || y == 0 || y == imageH -1) {
        // if we're on the edgejust set the pixel to be black
        d_output[i] = 0;
    } else {
        // apply edge detection filter here.  
        uint up = (y+1) * imageW + x;
        uint down = (y-1) * imageW + x;
        uint left = y * imageW + (x - 1);
        uint right = y * imageW + (x + 1);
        uint upRight = (y+1) * imageW + (x+1);
        uint upLeft = (y+1) * imageW + (x-1);
        uint downRight = (y-1) * imageW + (x+1);
        uint downLeft = (y-1) * imageW + (x-1);

        int indices [9] = {upLeft, up, upRight, left, i, right, downLeft, down, downRight};

        int rollingR = 0;
        int rollingG = 0;
        int rollingB = 0;
        int rollingA = 0;

        for (int j = 0; j < 9; j++) {
            uint colorI = d_input[indices[j]];
            uint aI = colorI >> 24 & 255; 
            uint rI = colorI >> 16 & 255; 
            uint gI = colorI >> 8 & 255; 
            uint bI = colorI >> 0 & 255; 
            if (j == 4) {
                rollingR += 8 * rI;
                rollingG += 8 * gI;
                rollingB += 8 * bI;
                rollingA += 8 * aI;
            } else {
                rollingR -= rI;
                rollingG -= gI;
                rollingB -= bI;
                rollingA -= aI;
            }
        }

        rollingR = min(max(int(1 * rollingR + 0), 0), 255); 
        rollingG = min(max(int(1 * rollingG + 0), 0), 255); 
        rollingB = min(max(int(1 * rollingB + 0), 0), 255); 

        uint newcolor = uint(255 << 24 | rollingR << 16 | rollingG << 8 | rollingB);
        d_output[i] = newcolor;
    }
}

// CPU version
void image_Blur(uint *d_input, uint imageW, uint imageH, uint *d_output, int blurSize)
{

    int *filter = (int *) malloc(blurSize * blurSize * sizeof(int));
    for (int k = 0; k < blurSize * blurSize; k++) {
        filter[k] = 1;
    }

    for (uint x = 0; x < imageW; ++x) {
        for (uint y = 0; y < imageH; ++y) {

        uint i = y * imageW + x;

        // Calculate screen coordinates based on indices
        if ((x >= imageW) || (y >= imageH)) return;

        int filterWidth = blurSize;
        int filterHeight = blurSize;

    
        int divideFactor = blurSize * blurSize; // note that this is the sum of factors in filter

        int red = 0;
        int blue = 0;
        int green = 0;
        for(int filterX = 0; filterX < filterWidth; filterX++) {
            for(int filterY = 0; filterY < filterHeight; filterY++) { 

                // note that we wrap around the edges/borders here with modulo
                int imageX = (x - filterWidth / 2 + filterX + imageW) % imageW; 
                int imageY = (y - filterHeight / 2 + filterY + imageH) % imageH; 

                // extract the color of the pixel...
                uint index = (imageY) * imageW + imageX;
                uint colorI = d_input[index];

                // and multiply RGB components according to filter
                uint r = colorI >> 16 & 255; 
                uint g = colorI >> 8 & 255; 
                uint b = colorI >> 0 & 255; 

                red += r * filter[3*filterY + filterX]; 
                green += g * filter[3*filterY + filterX]; 
                blue += b * filter[3*filterY + filterX];
            } 
        }

        // our average RGB from filter
        int rollingR = red / divideFactor;
        int rollingB = blue / divideFactor;
        int rollingG = green / divideFactor;

        // clamp between 0 and 255 before setting output color
        rollingR = min(max(int(1 * rollingR + 0), 0), 255); 
        rollingG = min(max(int(1 * rollingG + 0), 0), 255); 
        rollingB = min(max(int(1 * rollingB + 0), 0), 255); 

        uint newcolor = uint(255 << 24 | rollingR << 16 | rollingG << 8 | rollingB);
        d_output[i] = newcolor;
        }
    }
}


// GPU version
__global__ void d_image_Blur(uint *d_input, uint imageW, uint imageH, uint *d_output, int blurSize)
{
    // Calculate indices
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    extern __shared__ int filter[];

    if (threadIdx.x < blurSize && threadIdx.y < blurSize) {
        filter[blurSize * threadIdx.y + threadIdx.x] = 1;

    }

    __syncthreads();

    // Calculate screen coordinates based on indices
    if ((x >= imageW) || (y >= imageH)) return;

    uint i = y * imageW + x;

    int filterWidth = blurSize;
    int filterHeight = blurSize;

    
    int divideFactor = blurSize * blurSize; // note that this is the sum of factors in filter

    int red = 0;
    int blue = 0;
    int green = 0;
    for(int filterX = 0; filterX < filterWidth; filterX++) {
        for(int filterY = 0; filterY < filterHeight; filterY++) { 

            // note that we wrap around the edges/borders here with modulo
            int imageX = (x - filterWidth / 2 + filterX + imageW) % imageW; 
            int imageY = (y - filterHeight / 2 + filterY + imageH) % imageH; 

            // extract the color of the pixel...
            uint index = (imageY) * imageW + imageX;
            uint colorI = d_input[index];

            // and multiply RGB components according to filter
            uint r = colorI >> 16 & 255; 
            uint g = colorI >> 8 & 255; 
            uint b = colorI >> 0 & 255; 

            red += r * filter[3*filterY + filterX]; 
            green += g * filter[3*filterY + filterX];
            blue += b * filter[3*filterY + filterX];
        } 
    }

    // our average RGB from filter
    int rollingR = red / divideFactor;
    int rollingB = blue / divideFactor;
    int rollingG = green / divideFactor;

    // clamp between 0 and 255 before setting output color
    rollingR = min(max(int(1 * rollingR + 0), 0), 255); 
    rollingG = min(max(int(1 * rollingG + 0), 0), 255); 
    rollingB = min(max(int(1 * rollingB + 0), 0), 255); 

    uint newcolor = uint(255 << 24 | rollingR << 16 | rollingG << 8 | rollingB);
    d_output[i] = newcolor;
}

// GPU version
__global__ void d_image_Oil(uint *d_input, uint imageW, uint imageH, uint *d_output, int blurSize)
{
    // Calculate indices
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Calculate screen coordinates based on indices
    if ((x >= imageW) || (y >= imageH)) return;

    uint i = y * imageW + x;

    // we have 256 possible intensities to consider (0 to 255)
    int intensityCount[256];
    int avgR[256];
    int avgG[256];
    int avgB[256];

    // we'll consider 25 bins
    int numberBins = 25;
    float factor = 1.0 * numberBins / (3 * 255.0f);

    // Zero out count and average RGB's for each bin
   for (int b = 0; b <= numberBins; b++) {
        intensityCount[b] = 0;
        avgR[b] = 0;
        avgG[b] = 0;
        avgB[b] = 0;
    }

    int maxIntensityCount = 0;
    int maxIntensityCountIndex = 0;

    uint k = (y > blurSize) ? y - blurSize : 0;
  
    for (; k <= y + blurSize; k++) {
        if (k >= imageH) {
            continue; // don't consider if k is out of y-range of image
        }

        uint l = (x > blurSize) ? x - blurSize : 0;

        for (; l <= x + blurSize; l++) {
            if (l >= imageW) { // don't consider if l is out of x-range of image
                continue;
            }
            // grab RGB of each pixel
            uint index = k * imageW + l;
            uint colorI =  d_input[index];

            uint r = colorI >> 16 & 255; 
            uint g = colorI >> 8 & 255; 
            uint b = colorI >> 0 & 255; 

            // compute intensity of this pixel (average of RGB)
            uint curBin = (uint) ((r + g + b) * factor);

            intensityCount[curBin]++;
            if (intensityCount[curBin] > maxIntensityCount) {
                maxIntensityCount = intensityCount[curBin];
                maxIntensityCountIndex = curBin;
            }
            avgR[curBin] += r;
            avgG[curBin] += g;
            avgB[curBin] += b;
        }
    }


    int finalR = avgR[maxIntensityCountIndex] / maxIntensityCount;
    int finalG = avgG[maxIntensityCountIndex] / maxIntensityCount;
    int finalB = avgB[maxIntensityCountIndex] / maxIntensityCount;

    uint newcolor = (uint)(255 << 24 | finalR << 16 | finalG << 8 | finalB);
    d_output[i] = newcolor;
}

// CPU version
void image_Oil(uint *d_input, uint imageW, uint imageH, uint *d_output, int blurSize)
{
    for (int x = 0; x < imageW; x++) {
        for (int y = 0; y < imageH; y++) {

            uint i = y * imageW + x;

            // we have 256 possible intensities to consider (0 to 255)
            int intensityCount[256];
            int avgR[256];
            int avgG[256];
            int avgB[256];

            // we'll consider 25 bins
            int numberBins = 25;
            float factor = 1.0 * numberBins / (3 * 255.0f);

            // Zero out count and average RGB's for each bin
            for (int b = 0; b <= numberBins; b++) {
                intensityCount[b] = 0;
                avgR[b] = 0;
                avgG[b] = 0;
                avgB[b] = 0;
            }

            int maxIntensityCount = 0;
            int maxIntensityCountIndex = 0;

            uint k = (y > blurSize) ? y - blurSize : 0;
  
            for (; k <= y + blurSize; k++) {
                if (k >= imageH) {
                    continue; // don't consider if k is out of y-range of image
                }

                uint l = (x > blurSize) ? x - blurSize : 0;

                for (; l <= x + blurSize; l++) {
                    if (l >= imageW) { // don't consider if l is out of x-range of image
                        continue;
                    }
                
                    // grab RGB of each pixel
                    uint index = k * imageW + l;
                    uint colorI =  d_input[index];

                    uint r = colorI >> 16 & 255; 
                    uint g = colorI >> 8 & 255; 
                    uint b = colorI >> 0 & 255; 

                    // compute intensity of this pixel (average of RGB)
                    uint curBin = (uint) ((r + g + b) * factor);

                    intensityCount[curBin]++;
                    if (intensityCount[curBin] > maxIntensityCount) {
                        maxIntensityCount = intensityCount[curBin];
                        maxIntensityCountIndex = curBin;
                    }

                    avgR[curBin] += r;
                    avgG[curBin] += g;
                    avgB[curBin] += b;
                }
            }


            int finalR = avgR[maxIntensityCountIndex] / maxIntensityCount;
            int finalG = avgG[maxIntensityCountIndex] / maxIntensityCount;
            int finalB = avgB[maxIntensityCountIndex] / maxIntensityCount;

            uint newcolor = (uint)(255 << 24 | finalR << 16 | finalG << 8 | finalB);
            d_output[i] = newcolor;

        }
    }
}

// CPU
void image_Rotate(GLuint *d_input, uint imageW, uint imageH, GLuint *d_output)
{
    for (uint x = 0; x < imageW; ++x) {
        for (uint y = 0; y < imageH; ++y) {

            // calculate points in (U, V) space where (0,0) in UV is the center of the image
            double V = (0.5 * imageH) - y;
            double U = x - (0.5 * imageW);

            // Calculate the angle our points are relative to UV origin. Everything is in radians.
            double originalAngle;
            if (U != 0) {
                originalAngle = atan(abs(V)/abs(U));
                if (U > 0 && V < 0) {
                    originalAngle = 2.0f * C_PI - originalAngle;
                } else if (U <= 0 && V >= 0) {
                    originalAngle = C_PI-originalAngle;
                } else if (U <=0 && V < 0) {
                    originalAngle += C_PI;
                }
            }
            else
            {
                // Take care of rare special case
                if (V >= 0) {
                    originalAngle = 0.5f * C_PI;
                } else {
                    originalAngle = 1.5f * C_PI;
                }
            }
            // Calculate the distance from the center of the UV using pythagorean distance
            double radius = sqrt(U * U + V * V);
            double factor = 0.005;

            // we will increase the angle by a factor proportional to the radius.
            // the bigger the radius, the less we will swirl.
            double newAngle = originalAngle + 1 / (factor * radius + (4.0f/C_PI));

            // figure out where we must swirl from to get to our current pixel
            int oldX = (int)(floor(radius * cos(newAngle)+0.5f));
            int oldY = (int)(floor(radius * sin(newAngle)+0.5f));

            // convert back to pixel coordinates from UV
            oldX += (0.5 * imageW);
            oldY += (0.5 * imageH);
            oldY = imageH - oldY;

            // Clamp the source to legal image pixel
            if (oldX < 0) {
                oldX = 0;
            } else if (oldX >= imageW) {
                oldX = imageW-1;
            } 

            if (oldY < 0) {
                oldY = 0;
            }
            else if (oldY >= imageH) {
                oldY = imageH-1;
            }

            // Set the pixel color, as the input pixel from where we swirled from
            d_output[y * imageW + x] = d_input[oldY * imageW + oldX];
        }
    }
}

// GPU
__global__ void d_image_Rotate(uint *d_input, uint imageW, uint imageH, uint *d_output)
{
    // Calculate indices
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    // calculate points in (U, V) space where (0,0) in UV is the center of the image
    double V = (0.5 * imageH) - y;
    double U = x - (0.5 * imageW);

    // Calculate the angle our points are relative to UV origin. Everything is in radians.
    double originalAngle;
    if (U != 0)
    {
        originalAngle = atan(abs(V)/abs(U));
        if (U > 0 && V < 0) {
            originalAngle = 2.0f * C_PI - originalAngle;
        } else if (U <= 0 && V >=0) {
            originalAngle = C_PI-originalAngle;
        } else if (U <= 0 && V <0) {
            originalAngle += C_PI;
        } 
    }
    else
    {
        // Take care of rare special case
        if (V >= 0){
            originalAngle = 0.5f * C_PI;
        } else {
            originalAngle = 1.5f * C_PI;
        }
    }

    // Calculate the distance from the center of the UV using pythagorean distance
    double radius = sqrt(U * U + V * V);
    double factor = 0.005;

    // Use any equation we want to determine how much to rotate image by
    //double newAngle = originalAngle + factor*radius;  // a progressive twist
    double newAngle = originalAngle + 1 / (factor * radius + (4.0f/C_PI));

    // figure out where we must swirl from to get to our current pixel
    int oldX = (int)(floor(radius * cos(newAngle)+0.5f));
    int oldY = (int)(floor(radius * sin(newAngle)+0.5f));

    // convert back to pixel coordinates from UV
    oldX += (0.5 * imageW);
    oldY += (0.5 * imageH);
    oldY = imageH - oldY;

    // Clamp the source to legal image pixel
    if (oldX < 0) {
        oldX = 0;
    }
    else if (oldX >= imageW) {
        oldX = imageW-1;
    }

    if (oldY < 0) {
        oldY = 0;
    } else if (oldY >= imageH) {
        oldY = imageH-1;
    }

    // Set the pixel color, as the input pixel from where we swirled from
    d_output[y * imageW + x] = d_input[oldY * imageW + oldX];
}

void process_cpu(int width, int height, GLuint *in_image, GLuint *out_image, int drawMode, int gRadius)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    if (drawMode == 1) {
        memcpy(out_image, in_image, width * height * sizeof(unsigned int));
    } else if (drawMode == 2) {
        image_proc(in_image, width, height, out_image);
    } else if (drawMode == 3) {
        image_Rotate(in_image, width, height, out_image);
    } else if (drawMode == 4) {
        image_Edge(in_image, width, height, out_image);
    } else if (drawMode == 5) {
        image_Blur(in_image, width, height, out_image, gRadius);
    } else if (drawMode == 6) {
        image_Oil(in_image, width, height, out_image, gRadius);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Use cudaEvent_t's to calculate elapsed time, and clean up events
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Recompute for CPU method %d took %f ms\n", drawMode, elapsedTime);
}

/* Execute volume rendering kernel */
void process_gpu(int width, int height, GLuint *in_image, GLuint *out_image, int drawMode, int gRadius, int gRunTests)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    uint *d_input;
    cudaMalloc((void **) &d_input, width * height * sizeof(GLuint));
    cudaMemcpy(d_input, in_image, width * height * sizeof(GLuint), cudaMemcpyHostToDevice);

    uint *d_output;
    cudaMalloc((void **) &d_output, width * height * sizeof(GLuint));
    cudaMemcpy(d_output, in_image, width * height * sizeof(GLuint), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // Execute the kernel based on the draw mode
    for (int i = 0; i < gRunTests; i++) {
        if (drawMode == 2) {
            d_image_proc <<<gridSize, blockSize>>> (d_input, width, height, d_output);
        } else if (drawMode == 3) {
            d_image_Rotate <<<gridSize, blockSize>>> (d_input, width, height, d_output);
        } else if (drawMode == 4) {
            d_image_Edge <<<gridSize, blockSize>>> (d_input, width, height, d_output);
        } else if (drawMode == 5) {
            d_image_Blur <<<gridSize, blockSize, gRadius * gRadius * sizeof(int)>>> (d_input, width, height, d_output, gRadius);
        } else if (drawMode == 6) {
            d_image_Oil <<<gridSize, blockSize>>> (d_input, width, height, d_output, gRadius);
        }
    }

    cudaMemcpy(out_image, d_output, width * height * sizeof(GLuint), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Use cudaEvent_t's to calculate elapsed time, and clean up events
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Average recompute for GPU method %d took %f ms\n", drawMode, elapsedTime/gRunTests);
}

////////////////////////////////////////////////////////////////////////////////
//! End TODO list
////////////////////////////////////////////////////////////////////////////////

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

