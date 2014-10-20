/*
* Lab 4 - Fractal Volume Rendering
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <png.h>

#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_gl_interop.h>
#include "image_kernel.cuh"

bool gRunGPU = false;
/* Screen drawing kernel parameters */
int xRes = 512, yRes = 512;
int imageW;
int imageH;
float invViewMatrix[12];

/* View settings */
float3 viewRotation = make_float3(0.5, 0.5, 0.0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);

GLuint *gInData;
GLuint *gOutData;

int gDrawMode = 1;
int gRadius = 3;
int gRunTests = 1;

// display results using OpenGL (called by GLUT)
void display()
{
    if (gRunGPU) {
        process_gpu(imageW, imageH, gInData, gOutData, gDrawMode, gRadius, gRunTests);
    } else {
        process_cpu(imageW, imageH, gInData, gOutData, gDrawMode, gRadius);
    }

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);

    // draw the input image on the left hand side
    glDrawPixels(imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, gInData);


    // Set matrix mode
    glMatrixMode(GL_PROJECTION);
    // push current projection matrix on the matrix stack
    glPushMatrix();
    // Set an ortho projection based on window size
    glLoadIdentity();
    glOrtho(0, imageW * 2, 0, imageH, 0, 1);
    // Switch back to model-view matrix
    glMatrixMode(GL_MODELVIEW);
    // Store current model-view matrix on the stack
    glPushMatrix();
    // Clear the model-view matrix
    glLoadIdentity();
    // You can specify this in window coordinates now
    glRasterPos2i(imageW,0);
    glDrawPixels(imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, gOutData);
    // Restore the model-view matrix
    glPopMatrix();
    // Switch to projection matrix and restore it
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();



    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
}

void keyboard(unsigned char key, int x, int y)
{
    if (key == '0') {
      gDrawMode = 0;
    } else if (key == '1') {
      gDrawMode = 1;
    } else if (key == '2') {
      gDrawMode = 2;
    } else if (key == '3') {
      gDrawMode = 3;
    } else if (key == '4') {
      gDrawMode = 4;
    } else if (key == '5') {
      gDrawMode = 5;
    } else if (key == '6') {
      gDrawMode = 6;
    } else if (key =='c') {
      gRunGPU = false;
      std::cout << "Running CPU algorithms..." << std::endl;
      return;
    } else if (key == 'g') {
      gRunGPU = true;
      std::cout << "Running GPU algorithms..." << std::endl;
      return;
    } else if (key == '+') {
      if (gRadius < 14) {
        gRadius += 2;
      }
      std::cout << "Current radius for filters = " << gRadius << std::endl;
    } else if (key == '-') {
      if (gRadius > 2) {
        gRadius -= 2;
      }
      std::cout << "Current radius for filters = " << gRadius << std::endl;
    } else if (key == 27 || key == 'q') {
      exit(0);
    }

    glutPostRedisplay();
}

int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    glutPostRedisplay();
}

void reshape(int x, int y)
{
    xRes = x;
    yRes = y;

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

// function to load an PNG (specified by name) into outData.  Tells us the height, width, and
// whether there is an alpha channel or not
bool loadPngImage(char *name, int &outWidth, int &outHeight, bool &outHasAlpha, GLubyte **outData) {
   png_structp png_ptr;
   png_infop info_ptr;
   unsigned int sig_read = 0;
   int color_type, interlace_type;
   FILE *fp;

   if ((fp = fopen(name, "rb")) == NULL)
       return false;

   /* Create and initialize the png_struct
    * with the desired error handler
    * functions.  If you want to use the
    * default stderr and longjump method,
    * you can supply NULL for the last
    * three parameters.  We also supply the
    * the compiler header file version, so
    * that we know if the application
    * was compiled with a compatible version
    * of the library.  REQUIRED
    */
   png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                    NULL, NULL, NULL);

   if (png_ptr == NULL) {
       fclose(fp);
       return false;
   }

   /* Allocate/initialize the memory
    * for image information.  REQUIRED. */
   info_ptr = png_create_info_struct(png_ptr);
   if (info_ptr == NULL) {
       fclose(fp);
       png_destroy_read_struct(&png_ptr, NULL, NULL);
       return false;
   }

   /* Set error handling if you are
    * using the setjmp/longjmp method
    * (this is the normal method of
    * doing things with libpng).
    * REQUIRED unless you  set up
    * your own error handlers in
    * the png_create_read_struct()
    * earlier.
    */
   if (setjmp(png_jmpbuf(png_ptr))) {
       /* Free all of the memory associated
        * with the png_ptr and info_ptr */
       png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
       fclose(fp);
       /* If we get here, we had a
        * problem reading the file */
       return false;
   }

   /* Set up the output control if
    * you are using standard C streams */
   png_init_io(png_ptr, fp);

   /* If we have already
    * read some of the signature */
   png_set_sig_bytes(png_ptr, sig_read);

   /*
    * If you have enough memory to read
    * in the entire image at once, and
    * you need to specify only
    * transforms that can be controlled
    * with one of the PNG_TRANSFORM_*
    * bits (this presently excludes
    * dithering, filling, setting
    * background, and doing gamma
    * adjustment), then you can read the
    * entire image (including pixels)
    * into the info structure with this
    * call
    *
    * PNG_TRANSFORM_STRIP_16 |
    * PNG_TRANSFORM_PACKING  forces 8 bit
    * PNG_TRANSFORM_EXPAND forces to
    *  expand a palette into RGB
    */
   png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);

   png_uint_32 width, height;
   int bit_depth;
   png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
                &interlace_type, NULL, NULL);
   outWidth = width;
   outHeight = height;

   
       switch(color_type)
       {
               case PNG_COLOR_TYPE_RGBA:
                       outHasAlpha = true;
                       break;
               case PNG_COLOR_TYPE_RGB:
                       outHasAlpha = false;
                       break;
               default:
                       printf("%s: color type: %d not supported\n", __FILE__, color_type);
                       png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
                       fclose(fp);
                       return false;
       }



   unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
   *outData = (unsigned char*) malloc(row_bytes * outHeight);

   png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);

   for (int i = 0; i < outHeight; i++) {
       // note that png is ordered top to
       // bottom, but OpenGL expect it bottom to top
       // so the order or swapped
       memcpy(*outData+(row_bytes * (outHeight-1-i)), row_pointers[i], row_bytes);
   }

   /* Clean up after the read,
    * and free any memory allocated */
   png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

   /* Close the file */
   fclose(fp);

   /* That's it */
   return true;
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{

    if (argc == 2) {
      gRunTests = atoi(argv[1]);
    }

    GLubyte *image;
    bool hasAlpha;
    char *filename = "Lenna.png";
    bool success = loadPngImage(filename, imageW, imageH, hasAlpha, &image);
    if (success == false) {
        std::cerr << "Can't load " << filename << std::endl;
        exit(1);
    }

    gInData = new unsigned int [imageW * imageH];
    gOutData = new unsigned int [imageW * imageH];

    if (hasAlpha) {
      memcpy(gInData, image, imageW * imageH * sizeof(unsigned int));
    } else {
      GLubyte *inData = (GLubyte *)(gInData);
      for (int i = 0; i < imageW * imageH; i++) {
        inData[4*i] = image[3*i];
        inData[4*i + 1] = image[3*i + 1];
        inData[4*i + 2] = image[3*i + 2];
        inData[4*i + 3] = 255;
      }
    }
    delete image;

    // initialize output w/ input
    memcpy(gOutData, gInData, imageW * imageH * sizeof(unsigned int));

    printf("Press '+' and '-' to change radius of pixels considered for blur and oil painting \n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(imageW * 2, imageH);
    glutInitWindowPosition(500,300);
    glutCreateWindow("CUDA image processing!");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    //glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();

    glutMainLoop();

    return 0;
}
