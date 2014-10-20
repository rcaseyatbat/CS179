#include "interact_kernel.cuh"
#include <helper_math.h>
#include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// constants & defines
//TODO:: Choose one!
// Number of threads in a block.
#define BLOCK_SIZE 512

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

// Flag for pingpong;
int pingpong = 0;

unsigned int numBodies;     // Number particles; determined at runtime.

// Device buffer variables
float4* dVels[2];

__device__ float4 get_force(float4 pos, float4* neighbors)
{
    // TODO :: Calculate the acceleration, or force, on this particle
    //         See recitation slides or homework page for details.
    float4 accel = make_float4(0.0, 0.0, 0.0, 0.0);
    return accel;
}
__global__ void interact_kernel(float4* newPos, float4* oldPos, float4* newVel, 
                                float4* oldVel, float dt, float damping, 
                                int numBodies)
{
    // TODO:: Get unique thread id

    // TODO:: Load old positions into shared memory
    extern __shared__ float4 pos[];

    // TODO:: Calculate force using get_force function, then calculate the new 
    //        velocity
    
    // TODO:: Update position

    // TODO:: Write back to old values
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(GLuint *vbo, float dt)
{
    // map OpenGL buffer object for writing from CUDA
    float4* oldPos;
    float4* newPos;

    // Velocity damping factor, using it is optional though.
    float damping = 0.995;

    // TODO:: Map opengl buffers to CUDA.

    // TODO:: Choose a block size, a grid size, an amount of shared mem,
    // and execute the kernel
    // dVels is the particle velocities old, new. Pingponging of these is
    // handled, if the initial conditions have initial velocities in dVels[0].

    // TODO:: unmap buffer objects from cuda.

    // TODO:: Switch buffers between old/new
}

////////////////////////////////////////////////////////////////////////////////
//! Create device data
////////////////////////////////////////////////////////////////////////////////
void createDeviceData()
{
    gpuErrchk(cudaMalloc((void**)&dVels[0], numBodies *
                                                    4 * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&dVels[1], numBodies *
                                                    4 * sizeof(float)));

    // Initialize data.
    float4* tempvels = (float4*)malloc(numBodies * 4*sizeof(float));
    for(int i = 0; i < numBodies; ++i)
    {
        // TODO: set initial velocity data
        tempvels[i].x = 0.f;
        tempvels[i].y = 0.f;
        tempvels[i].z = 0.f;
        tempvels[i].w = 1.f;
    }

    // Copy to gpu
    gpuErrchk(cudaMemcpy(dVels[0], tempvels, numBodies*4*sizeof(float), cudaMemcpyHostToDevice));

    free(tempvels);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBOs(GLuint* vbo)
{
    // create buffer object
    glGenBuffers(2, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

    // initialize buffer object; this will be used as 'oldPos' initially
    unsigned int size = numBodies * 4 * sizeof(float);

    float4* temppos = (float4*)malloc(numBodies*4*sizeof(float));
    for(int i = 0; i < numBodies; ++i)
    {
        // TODO :: Modify initial positions!
        temppos[i].x = 0.;
        temppos[i].y = 0.;
        temppos[i].z = 0.;
        temppos[i].w = 1.;
    }

    // Notice only vbo[0] has initial data!
    glBufferData(GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    free(temppos);

    // Create initial 'newPos' buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);


    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register buffer objects with CUDA
    gpuErrchk(cudaGLRegisterBufferObject(vbo[0]));
    gpuErrchk(cudaGLRegisterBufferObject(vbo[1]));
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBOs(GLuint* vbo)
{
    glBindBuffer(1, vbo[0]);
    glDeleteBuffers(1, &vbo[0]);
    glBindBuffer(1, vbo[1]);
    glDeleteBuffers(1, &vbo[1]);

    gpuErrchk(cudaGLUnregisterBufferObject(vbo[0]));
    gpuErrchk(cudaGLUnregisterBufferObject(vbo[1]));

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Delete device data
////////////////////////////////////////////////////////////////////////////////
void deleteDeviceData()
{
    // Create a velocity for every position.
    gpuErrchk(cudaFree(dVels[0]));
    gpuErrchk(cudaFree(dVels[1]));
    // pos's are the VBOs
}

////////////////////////////////////////////////////////////////////////////////
//! Returns the value of pingpong
////////////////////////////////////////////////////////////////////////////////
int getPingpong()
{
  return pingpong;
}

////////////////////////////////////////////////////////////////////////////////
//! Gets/sets the number of bodies
////////////////////////////////////////////////////////////////////////////////
int getNumBodies()
{
  return numBodies;
}
void setNumBodies(int n)
{
  numBodies = n;
}
