#ifndef IMAGE_KERNEL_CUH
#define IMAGE_KERNEL_CUH

#include <cuda_runtime.h>
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif

#include <cuda_gl_interop.h>

void process_gpu(int width, int height, GLuint *in_image, GLuint *out_image, int gDrawMode, int gRadius, int gRunTests);
void process_cpu(int width, int height, GLuint *in_image, GLuint *out_image, int gDrawMode, int gRadius);

int iDivUp(int a, int b);
#endif // IMAGE_KERNEL_CUH
