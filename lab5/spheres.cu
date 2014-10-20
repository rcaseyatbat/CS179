// Lab 5 - volume of union of spheres

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include <cublas.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_math.h>

#include "SimpleRNG.h"

static const int NUM_PTS = 1024 * 1024;
static const int NUM_SPHERES = 128;

// spheres represented with x, y, z and posn and w as radius

// macro for error-checking CUDA calls
#define gpuErrchk(ans) { gpuAssert((ans), (char*)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

////////////////////////////////////////////////////////////////////////////////
// helper functions for this lab...

typedef struct {
  double xmin, xmax, ymin, ymax, zmin, zmax;
  double xrange, yrange, zrange;
  double volume;
} BoundBox;

// find the bounding box for a set of spheres
void FindBoundingBox(float4* spheres, int numSpheres, BoundBox& box) {
  box.xmin = box.xmax = spheres[0].x;
  box.ymin = box.ymax = spheres[0].y;
  box.zmin = box.zmax = spheres[0].z;
  for (int x = 0; x < numSpheres; x++) {
    if (box.xmin > spheres[x].x - spheres[x].w)
      box.xmin = spheres[x].x - spheres[x].w;
    if (box.ymin > spheres[x].y - spheres[x].w)
      box.ymin = spheres[x].y - spheres[x].w;
    if (box.zmin > spheres[x].z - spheres[x].w)
      box.zmin = spheres[x].z - spheres[x].w;
    if (box.xmax < spheres[x].x + spheres[x].w)
      box.xmax = spheres[x].x + spheres[x].w;
    if (box.ymax < spheres[x].y + spheres[x].w)
      box.ymax = spheres[x].y + spheres[x].w;
    if (box.zmax < spheres[x].z + spheres[x].w)
      box.zmax = spheres[x].z + spheres[x].w;
  }
  box.xrange = box.xmax - box.xmin;
  box.yrange = box.ymax - box.ymin;
  box.zrange = box.zmax - box.zmin;
  box.volume = box.xrange * box.yrange * box.zrange;
}

// return the current time, in seconds
double now() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000;
}

// generate a "random" seed based on time
long long random_seed() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Begin TODO                                                                //
////////////////////////////////////////////////////////////////////////////////
// check if a point is inside a sphere
__device__ __host__ bool PointInSphere(float3& pt, float4& sphere) {
    // TODO: check if the point 'pt' is in 'sphere'
    if ((pt.x-sphere.x)*(pt.x-sphere.x) + (pt.y-sphere.y)*(pt.y-sphere.y) + 
        (pt.z-sphere.z)*(pt.z-sphere.z) <= (sphere.w * sphere.w)) {
        return true;
    } else {
        return false;
    }
}

SimpleRNG rng;

////////////////////////////////////////////////////////////////////////////////
// kernels

// inputs:
//   spheres, numSpheres - describe the array of spheres
//   points - points to check against spheres; coordinates are in [0, 1]^3
//   doubleResults, intResults - arrays of doubles and floats to write results
//     to. either can be NULL, in which case results aren't written to them
//   box - bounding box to scale points into
// total number of threads must be equal to the number of points
__global__ void CheckPointsK(float4* spheres, int numSpheres, float3* points,
    double* doubleResults, unsigned int* intResults, BoundBox box) {

  // TODO: check if the point is inside any sphere. if so, set the appropriate
  //       entry in doubleResults and intResults to 1 (if non-NULL).
    extern __shared__ float4 spheresCopy[];

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < numSpheres) {
        spheresCopy[threadIdx.x] = spheres[threadIdx.x];
    }
    syncthreads();

    float3 scaledPoint;
    float3 point = points[index];
    scaledPoint.x = point.x * box.xrange + box.xmin;
    scaledPoint.y = point.y * box.yrange + box.ymin;
    scaledPoint.z = point.z * box.zrange + box.zmin;

    
    if (intResults) {
        intResults[index] = 0;
    }
    if (doubleResults) {
        doubleResults[index] = 0.0;
    }
    
	//syncthreads();

    for (int i = 0; i < numSpheres; i++) {
        if (PointInSphere(scaledPoint, spheresCopy[i])) {
            if (intResults) {
                intResults[index] = 1;
            }
            if (doubleResults) {
                doubleResults[index] = 1.0;
            }
        }
    }
    

}

// generates 'count' random float3s using CURAND
// only requires the total number of threads to be a factor of 'count'
// ex. can call as such: GenerateRandom3K<<< 3, 8 >>>(..., 72)
__global__ void GenerateRandom3K(float3* toWrite, long long seed,
                                 curandState* states, int count) {
  // TODO: initialize 'count' many random generator states, then generate 
  // random float3s into 'toWrite[]'
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    //curand_init(seed, index, 0, &states[index]);
    for (unsigned int x = index; x < count; x += (blockDim.x * gridDim.x)) {     
        curand_init(seed, x, 0, &states[x]);
        toWrite[x] = make_float3(curand_uniform(&states[x]),curand_uniform(&states[x]),curand_uniform(&states[x]));
    }
}

__global__ void SumVector(unsigned int *vector, int size)
{
    // TODO: add a reduction kernel to sum an array of unsigned ints
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        vector[index] += vector [index + size];
    }
    
}

////////////////////////////////////////////////////////////////////////////////
// host code

// find volume on GPU, summing using CUBLAS
double VolumeCUBLAS(float4* d_spheres, int numSpheres, int numPts,
                   BoundBox& box) {

    double vol = 0.0;

  // TODO: allocate memory for needed data
    float3 *dev_points;
    cudaMalloc((void**) &dev_points, numPts * sizeof(float3));

    double* dev_double;
    cudaMalloc((void**) &dev_double, numPts * sizeof(double));
    cudaMemset(dev_double, 0, numPts * sizeof(double));

    unsigned int* dev_int;
    cudaMalloc((void**) &dev_int, numPts * sizeof(unsigned int));
    cudaMemset(dev_int, 0, numPts * sizeof(unsigned int));

  // TODO: generate random points on GPU in [0, 1]^3 using CURAND host API
    curandGenerator_t r;
    curandCreateGenerator(&r, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(r, random_seed());
    curandGenerateUniform(r, (float *) dev_points, 3*numPts);
    curandDestroyGenerator(r);

  // TODO: check if each point is within any sphere

    const unsigned int threadsPerBlock = 512;
    const unsigned int blocks = ceil(numPts/(float) threadsPerBlock);

    CheckPointsK<<<blocks, threadsPerBlock, numSpheres * sizeof(float4)>>> (d_spheres, 
        numSpheres, dev_points, dev_double, dev_int, box);

  // TODO: count points using CUBLAS
    vol = box.volume * cublasDasum(numPts, dev_double, 1) / numPts;
    printf("Sum: %f \n", cublasDasum(numPts, dev_double, 1));

  // TODO: free memory on GPU
    cudaFree(dev_points);
    cudaFree(dev_double);
    cudaFree(dev_int);

    return vol;
}

// find volume on GPU, summing using reduction kernel
double VolumeCUDA(float4* d_spheres, int numSpheres, int numPts, BoundBox& box) {

    double vol = 0.0;

  // TODO: allocate memory for needed data (including random generator states)
    curandState *dev_states;
    cudaMalloc(&dev_states, numPts * sizeof(curandState));

    float3 *dev_points;
    cudaMalloc((void**) &dev_points, numPts * sizeof(float3));

    double* dev_double;
    cudaMalloc((void**) &dev_double, numPts * sizeof(double));
    cudaMemset(dev_double, 0, numPts * sizeof(double));

    unsigned int* dev_int;
    cudaMalloc((void**) &dev_int, numPts * sizeof(unsigned int));
    cudaMemset(dev_int, 0, numPts * sizeof(unsigned int));

  // TODO: generate random points on GPU in [0, 1]^3 using CURAND device API
    const unsigned int threadsPerBlock = 512;
    const unsigned int blocks = ceil(numPts/(float) threadsPerBlock);
    GenerateRandom3K<<<32, threadsPerBlock>>> (dev_points, random_seed(), dev_states, numPts);

    printf("PRESYNCH \n");
    cudaDeviceSynchronize(); // make sure one the first kernel finishes before we try to use its results
    printf("POST SYNCH \n");
  // TODO: check if each point is within any sphere
    CheckPointsK<<<blocks, threadsPerBlock, numSpheres * sizeof(float4)>>> 
        (d_spheres, numSpheres, dev_points, dev_double, dev_int, box);

    cudaDeviceSynchronize(); // make sure one the first kernel finishes before we try to use its results

  // TODO: count points using reduction kernel
    //vol = box.volume * cublasDasum(numPts, dev_double, 1) / numPts;
    //printf("Sum: %f \n", box.volume * cublasDasum(numPts, dev_double, 1) / numPts);

    int maxSize = numPts / 2; // note: require that numPts is a power of 2
    for (int i = maxSize; i > 0; i /= 2) {
        // call the reduction kernel on size  i
        SumVector <<<blocks, threadsPerBlock>>> (dev_int, i);
    }

    unsigned int *reducedSum = (unsigned int *)malloc(sizeof(unsigned int));
    cudaMemcpy(reducedSum, dev_int, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    vol = box.volume * reducedSum[0] / numPts;
    

  // TODO: free memory on GPU
    cudaFree(dev_states);
    cudaFree(dev_points);
    cudaFree(dev_double);
    cudaFree(dev_int);

    return vol;
}

////////////////////////////////////////////////////////////////////////////////
// End TODO                                                                   //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// main program

// find volume on CPU
double VolumeCPU(float4* spheres, int numSpheres, int numPts, BoundBox& box) {
  int x, y, numPtsInside = 0;
  for (x = 0; x < numPts; x++) {
    float3 pt = make_float3(rng.GetUniform() * box.xrange + box.xmin,
                            rng.GetUniform() * box.yrange + box.ymin,
                            rng.GetUniform() * box.zrange + box.zmin);
    for (y = 0; y < numSpheres; y++)
      if (PointInSphere(pt, spheres[y]))
        break;
    if (y < numSpheres)
      numPtsInside++;
  }

  return (double)numPtsInside / (double)numPts * box.volume;
}

void RunVolume(const char* name, double (*Vol)(float4*, int, int, BoundBox&),
    float4* spheres, int numSpheres, int numPts, BoundBox& box) {
  printf("find volume (%s)...\n", name);
  double start_time = now();
  double volume = Vol(spheres, numSpheres, numPts, box);
  double end_time = now();
  printf("  volume: %g\n", volume);
  printf("  time: %g sec\n", end_time - start_time);
}

int main(int argc, char** argv) {

  // seed the CPU random generator
  rng.SetState(random_seed(), random_seed());

  // set program parameters and allocate memory for spheres
  printf("generate spheres...\n");
  int numPts = NUM_PTS;
  int numSpheres = NUM_SPHERES;
  if (argc == 3) {
    numPts = atoi(argv[1]);
    numSpheres = atoi(argv[2]); // fixed typo here!
  }
  float4* spheres = (float4*)malloc(numPts * sizeof(float4));
  if (!spheres) {
    printf("failed to allocate memory for spheres\n");
    return -1;
  }

  // generate random spheres centered in [0, 10]^3
  double totalVolume = 0.0f;
  for (int x = 0; x < numSpheres; x++) {
    spheres[x].x = rng.GetUniform() * 10.0f;
    spheres[x].y = rng.GetUniform() * 10.0f;
    spheres[x].z = rng.GetUniform() * 10.0f;
    spheres[x].w = rng.GetUniform() + 1.0f;
    totalVolume += (4.0f * spheres[x].w * spheres[x].w * spheres[x].w * M_PI
                    / 3.0f);
    // uncomment to print spheres
    //printf("  sphere: (%g, %g, %g) with r = %g\n", spheres[x].x, spheres[x].y,
    //       spheres[x].z, spheres[x].w);
  }
  printf("  number of spheres: %u\n", numSpheres);
  printf("  non-union volume: %g\n", totalVolume);
  printf("  number of points: %u\n", numPts);

  // find bounding box of spheres
  printf("find bounds rect...\n");
  BoundBox box;
  FindBoundingBox(spheres, numSpheres, box);
  printf("  boundsrect: [%g, %g] x [%g, %g] x [%g, %g]\n", box.xmin, box.xmax,
         box.ymin, box.ymax, box.zmin, box.zmax);
  printf("  boundsrange: %g, %g, %g (volume %g)\n", box.xrange, box.yrange,
         box.zrange, box.volume);

  // init cublas and allocate memory on the GPU
  printf("initialize GPU...\n");
  cublasInit();
  float4* d_spheres;
  gpuErrchk(cudaMalloc(&d_spheres, numSpheres * sizeof(float4)));

  // copy the spheres to the GPU
  cudaMemcpy(d_spheres, spheres, numSpheres * sizeof(float4),
             cudaMemcpyHostToDevice);

  // run CPU version
  RunVolume("CPU", VolumeCPU, spheres, numSpheres, numPts, box);
  RunVolume("CUBLAS", VolumeCUBLAS, d_spheres, numSpheres, numPts, box);
  RunVolume("no CUBLAS", VolumeCUDA, d_spheres, numSpheres, numPts, box);

  // get rid of stuff in memory
  printf("clean up...\n");
  gpuErrchk(cudaFree(d_spheres));
  cublasShutdown();

  cudaThreadExit();
  return 0;
}

