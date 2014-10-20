/*
 * Lab 4 - Volume Rendering Fractals
 */

#ifndef _TEXTURE3D_KERNEL_H_
#define _TEXTURE3D_KERNEL_H_

#include "helper_math.h"
#include "frac3d_kernel.cuh"
#include <helper_cuda.h>
#include <stdio.h>
#include <cstdlib>

typedef unsigned int uint;

#define CUBE_DIM 256

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
#define WRAP(x,m) ((x)<(m)?(x):((x)-(m)))


cudaExtent volumeSize = make_cudaExtent(CUBE_DIM,CUBE_DIM,CUBE_DIM);

/* Recompute kernel parameters */
dim3 volBlockSize(volumeSize.width);
dim3 volGridSize(volumeSize.height, volumeSize.depth);

/* Recalculate next frame? */
bool recompute = true;

// Pointer to "volume texture" cudaArray
cudaArray *d_volumeArray = 0;
// Pointer to global memory array
float *d_volumeData = NULL;
// Pitched pointer and params
cudaPitchedPtr d_volumePtr = {0};
cudaMemcpy3DParms d_volumeParams = {0};

/* Volume texture declaration */
texture<float, 3, cudaReadModeElementType> tex;

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

/* Need to write host code to set these */
__constant__ float4 c_juliaC; // julia set constant
__constant__ float4 c_juliaPlane; // plane eqn of 3D slice


struct Ray {
    float3 o;	// origin
    float3 d;	// direction
};

// multiply two quaternions
    __device__ float4
mul_quat(float4 p, float4 q)
{
    return make_float4(p.x*q.x-p.y*q.y-p.z*q.z-p.w*q.w,
            p.x*q.y+p.y*q.x+p.z*q.w-p.w*q.z,
            p.x*q.z-p.y*q.w+p.z*q.x+p.w*q.y,
            p.x*q.w+p.y*q.z-p.z*q.y+p.w*q.x);
}

// square a quaternion (could be optimized)
    __device__ float4
sqr_quat(float4 p)
{
    // this could/should be optimized
    return mul_quat(p,p);
}

// convert a 3d position to a 4d quaternion using plane-slice
    __device__ float4
pos_to_quat(float3 pos, float4 plane)
{
    return make_float4(pos.x, pos.y, pos.z,
            plane.x*pos.x+plane.y*pos.y+plane.z*pos.z+plane.w);
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

    __device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
    __device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
    __device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

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

////////////////////////////////////////////////////////////////////////////////
//! TODO: Modify and fill in the code below where marked with a TODO tag.
////////////////////////////////////////////////////////////////////////////////

// computes julia distance function
__device__ float d_JuliaDist(float3 pos, int niter)
{
    int i = 0;
    
    // z_n: convert the position to a quaternion with the plane slice
    float4 z = pos_to_quat(pos, c_juliaPlane);

    // z_n' in the Julia set equation
    float4 zPrime = make_float4(1,0,0,0);  // z'_0 = (1,0,0,0)

    // break out if we reach number of iterations, or if |z_n|^2 > 20
    float zSquared = z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w;
    while( i < niter && zSquared <= 20.0f) // break if |z_n|^2 > 20
    {
         // Julia set equation
         zPrime = 2.0f * mul_quat(z, zPrime);  // z’_n+1= 2 *z_n *z’_n,
         // iterative complex equation
         z = sqr_quat(z) + c_juliaC; // z_n+1 = (z_n)^2 + c

         zSquared = z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w; // new value of z_n^2
   
         i++;
    }

    // magnitude is sqrt of sum of squares of components
    float magZ = sqrt(z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w);
    float magZPrime = sqrt(zPrime.x * zPrime.x + zPrime.y * zPrime.y + zPrime.z * zPrime.z + zPrime.w * zPrime.w);

    // calculate the distance using the formula from the slides
    float distance = (magZ / (2.0f * magZPrime)) * float(log(magZ));

    return distance;

}

// returns normal by sampling texture
__device__ float3 d_JuliaNormal(float3 pos)
{
    float3 normal = make_float3(1, 0, 0);

    float delta = 0.01;

    // grab pos + delta in every direction
    float3 newposXPlus = make_float3(pos.x + delta, pos.y, pos.z);
    float3 newposYPlus = make_float3(pos.x, pos.y + delta, pos.z);
    float3 newposZPlus = make_float3(pos.x, pos.y, pos.z + delta);

    // grab pos - delta in every direction
    float3 newposXMinus = make_float3(pos.x - delta, pos.y, pos.z);
    float3 newposYMinus = make_float3(pos.x, pos.y - delta, pos.z);
    float3 newposZMinus = make_float3(pos.x, pos.y, pos.z - delta);

    // calculate the components of the normal
    // for each direction, this is just the gradient of JuliaDist for the offset positions
    normal.x = (d_JuliaDist(newposXPlus, 100) - d_JuliaDist(newposXMinus, 100))/(delta);
    normal.y = (d_JuliaDist(newposYPlus, 100) - d_JuliaDist(newposYMinus, 100))/(delta);
    normal.z = (d_JuliaDist(newposZPlus, 100) - d_JuliaDist(newposZMinus, 100))/(delta);

    return normal;
}

// perform volume rendering
__global__ void d_render(uint *d_output, uint imageW, uint imageH, float epsilon)
{
    // amount to step by -- you can change these if desired
    float tfactor = 0.1;
    int maxSteps = 2000;

    // Calculate indices
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    // Calculate screen coordinates based on indices
    if ((x >= imageW) || (y >= imageH)) return;

    // convert to [-1, 1] space
    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = mul(c_invViewMatrix, normalize(make_float3(u, v, -2.0f)));

    // find intersection with box
    float tnear, tfar;

    // return if not intersecting
    if (!intersectBox(eyeRay, 
                make_float3(-2.0f,-2.0f,-2.0f), 
                make_float3(2.0f,2.0f,2.0f),
                &tnear, &tfar))
        return;

    // clamp to near plane
    if (tnear < 0.0f) tnear = 0.0f;

    float t = tnear;

    float3 pos;

    bool hit = false;

    for(int i=0; i<maxSteps; i++) {		
        // Raytrace to find point of intersection, store in pos
        pos = eyeRay.o + eyeRay.d * t;

        pos = pos*0.25f+0.5f;    // map position to [0,1]
        float distance = tex3D(tex, pos.x, pos.y, pos.z);
        if (distance <= epsilon) {
            hit = true;
            break;
        }

        t += tfactor*distance;

        if (t > tfar) {
            break;
        }
    }

    if (hit == false) {
        return; // we didn't hit anything
    }

    float3 Light = make_float3(1.0,1.0,1.0); // hardcoded Light location


    // EXTRA CREDIT - SHADOWS
    // Implemented by raytracing from position we hit towards the light
    Ray shadowRay;
    shadowRay.o = eyeRay.o + eyeRay.d * t; // get position that we hit 
    shadowRay.d = Light - pos; // now, start raytracing towards Light
    bool shadowHit = false;

    float shadow_t = tnear;
    for(int i=0; i<maxSteps; i++) {
        pos = shadowRay.o + (shadowRay.d)*shadow_t;
        pos = pos*0.25f+0.5f;    // map position to [0,1]

        float distance = tex3D(tex, pos.x, pos.y, pos.z);
        if (distance <= epsilon) {
            shadowHit = true;
            break;
        }

        shadow_t += tfactor*distance;
        if (shadow_t > tfar) {
            break;
        }
    }

    // grab normal vector of position that we hit, which we will use for lighting
    pos = eyeRay.o + eyeRay.d*t;
    float3 normal = d_JuliaNormal(pos);
    normal = normalize(normal);


    if ((x < imageW) && (y < imageH) && t < tfar) {
        uint i = y * imageW + x;

        float4 col4;

        // calculate output color based on lighting and position
        // Doing phong lighting

        float3 Ambient = make_float3(0.5, 0.5, 0.8);
        float3 Diffuse = make_float3(0.3, 0.3, 0.8);
        float3 Specular = make_float3(0.02, 0.02, 0.02);


        Light = normalize(Light); // make sure we use normalized vectors!
        float3 EyeVector = -1 * eyeRay.d;
        EyeVector = normalize(EyeVector);

        // dot(N, L) = |N| |L| cos(N, L). But |N| = |L| = 1 since normalized, 
        // so cos(N,L) = dot(N, L)
        float cosA = dot(normal, Light); // used for diffuse lighting


        // for specular lighting
        float3 R = 2*(dot(Light, normal))*normal - Light; // Refecltion of Light across Normal
        float shininess = 0.01; // shiny factor
        float RdotEye = dot(EyeVector, R);
        float specularFactor = pow(RdotEye, shininess);

        // sum together ambient, diffuse, and specular components of lighting
        float3 color = Ambient + (Diffuse * cosA) + (Specular * specularFactor);

        col4 = make_float4(color, 1.0f);

        if (shadowHit) { // make everything a little darker if we're in the shadow!
            col4.x = col4.x * 0.8;
            col4.y = col4.y * 0.8;
            col4.z = col4.z * 0.8;
        }

        d_output[i] = rgbaFloatToInt(col4);
    }
}

// recompute julia set at a single volume point
 __global__ void d_setfractal(float *d_output)
{
    // Get x,y,z indices from based on kernel block/grid architecture
    uint x = threadIdx.x;
    uint y = blockIdx.x;
    uint z = blockIdx.y;

    // map position to [-2,2] box
    float posX = ((float(x) / float(blockDim.x)) - 0.5) * 4.0;
    float posY = ((float(y) / float(gridDim.x)) - 0.5) * 4.0;
    float posZ = ((float(z) / float(gridDim.y)) - 0.5) * 4.0;

    float3 position = make_float3(posX, posY, posZ);

    // Calculate index based on indices.
    // Note: here we are using coalesced memory, which is faster
    long i = x + (gridDim.x * y) + (gridDim.x * gridDim.y * z);

    // Set d_output based on the results of d_JuliaDist.
    //       Remember to call d_JuliaDist with the right coordinates!
    float distance = d_JuliaDist(position, 100);
    d_output[i] = distance;
    
}

/* Execute the recompute kernel */
void recalculate()
{
    // Create timer events and start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // note: want to write to d_volumeData 
    // call d_setfractal kernel 
    d_setfractal<<<volGridSize, volBlockSize >>>(d_volumeData);

    // copy global memory to texture array
    checkCudaErrors(cudaMemcpy3D(&d_volumeParams));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Use cudaEvent_t's to calculate elapsed time, and clean up events
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Recompute took %f ms\n", elapsedTime);
}

/* Execute volume rendering kernel */
void render(GLuint pbo, int width, int height, float epsilon, float* invViewMatrix)
{
    // Copy inverse view matrix to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float4)*3));

    // If we need to recompute (recompute == true), call recalculate().
    //       Remember to let it know we don't need to recompute after this!
    if (recompute == true) {
        recalculate();
        recompute = false;
    }

    // Map pbo to get CUDA device pointer and zero the memory
    uint *d_output;
    checkCudaErrors(cudaGLMapBufferObject((void **)&d_output, pbo));
    cudaMemset(d_output, 0, width * height * sizeof(unsigned int));   // since we store an unsigned int for each pixel

    dim3 blockSize(16, 16);
    dim3 gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
    // Execute volume rendering kernel
    d_render <<<gridSize, blockSize>>> (d_output, width, height, epsilon);

    // Unmap the buffer
    checkCudaErrors(cudaGLUnmapBufferObject(pbo));
}

void setConsts(float4 juliaC, float4 juliaPlane)
{
    // Set c_juliaC and c_juliaPlane constants on device
    checkCudaErrors(cudaMemcpyToSymbol(c_juliaC, &juliaC, sizeof(float4)));
    checkCudaErrors(cudaMemcpyToSymbol(c_juliaPlane, &juliaPlane, sizeof(float4)));
}

////////////////////////////////////////////////////////////////////////////////
//! End TODO list
////////////////////////////////////////////////////////////////////////////////

void initCuda(int width, int height)
{
    int totalSize = sizeof(float)*volumeSize.width*volumeSize.height*volumeSize.depth;
    printf("total size: %d\n", totalSize);

    // create 3d cudaArray (in texture memory)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    gpuErrchk( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

    // create writeable 3d data array (in global memory)
    gpuErrchk( cudaMalloc((void**)&d_volumeData, totalSize));
    d_volumePtr = make_cudaPitchedPtr((void*)d_volumeData, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);

    // set up copy params for future copying
    d_volumeParams.srcPtr   = make_cudaPitchedPtr((void*)d_volumeData, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
    d_volumeParams.dstArray = d_volumeArray;
    d_volumeParams.extent   = volumeSize;
    d_volumeParams.kind     = cudaMemcpyDeviceToDevice;

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    gpuErrchk(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

void cleanup(GLuint pbo)
{
    gpuErrchk(cudaFree(d_volumeData));
    gpuErrchk(cudaFreeArray(d_volumeArray));
    gpuErrchk(cudaGLUnregisterBufferObject(pbo));    
    glDeleteBuffersARB(1, &pbo);
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer(GLuint *pbo, int width, int height)
{
    if (*pbo) {
        // delete old buffer
        gpuErrchk(cudaGLUnregisterBufferObject(*pbo));
        glDeleteBuffersARB(1, pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    gpuErrchk(cudaGLRegisterBufferObject(*pbo));
}

void setRecompute() { recompute = true; }
#endif // #ifndef _TEXTURE3D_KERNEL_H_
