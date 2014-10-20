// -*- C++ -*-
#ifndef MAIN1_CUDA_CUH
#define MAIN1_CUDA_CUH

#include <cuda_runtime.h>
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <cuda_gl_interop.h>

void runCuda(GLuint *vbo, float dt);
void createDeviceData();
void createVBOs(GLuint* vbo);
void deleteVBOs(GLuint* vbo);
void deleteDeviceData();
int getPingpong();
int getNumBodies();
void setNumBodies(int n);

#endif // MAIN1_CUDA_CUH
