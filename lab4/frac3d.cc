/*
* Lab 4 - Fractal Volume Rendering
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_gl_interop.h>
#include "frac3d_kernel.cuh"

GLuint pbo = 0;
/* Screen drawing kernel parameters */
int width = 512, height = 512;
float invViewMatrix[12];

/* Local Julia Set parameters */
float4 juliaC;
float4 juliaPlane;

/* Volume rendering isosurface */
float epsilon = 0.003f;


/* View settings */
float3 viewRotation = make_float3(0.5, 0.5, 0.0);
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);

// display results using OpenGL (called by GLUT)
void display()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
        glLoadIdentity();
        glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    // transpose matrix to conform with OpenGL's notation
    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];


    float distToCam = sqrt(viewTranslation.x * viewTranslation.x + 
                           viewTranslation.y * viewTranslation.y + 
                           viewTranslation.z * viewTranslation.z);


    // EXTRA CREDIT: ADAPTIVE SAMPLING
    render(pbo, width, height, epsilon * distToCam / 4.0, invViewMatrix);

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
}

void keyboard(unsigned char key, int x, int y)
{
    float ds = 0.005f;

    switch(key) {
        case 27:
            exit(0);
            break;
            break;
        case 'w':
            juliaC.y += ds;
            break;
        case 's':
            juliaC.y -= ds;
            break;
        case 'a':
            juliaC.x -= ds;
            break;
        case 'd':
            juliaC.x += ds;
            break;
        case 'q':
            juliaC.z -= ds;
            break;
        case 'e':
            juliaC.z += ds;
            break;
        case 'z':
            juliaC.w -= ds;
            break;
        case 'x':
            juliaC.w += ds;
            break;

        case 'i':
            juliaPlane.y += ds;
            break;
        case 'k':
            juliaPlane.y -= ds;
            break;
        case 'j':
            juliaPlane.x -= ds;
            break;
        case 'l':
            juliaPlane.x += ds;
            break;
        case 'u':
            juliaPlane.z -= ds;
            break;
        case 'o':
            juliaPlane.z += ds;
            break;
        case 'm':
            juliaPlane.w -= ds;
            break;
        case ',':
            juliaPlane.w += ds;
            break;


        case '=':
        case '+':
            epsilon *= 1.2;
            printf("epsilon = %.5f\n", epsilon);
            break;
        case '-':
            epsilon /= 1.2;
            printf("epsilon = %.5f\n", epsilon);
            break;

        default:
            break;
    }
    setRecompute();

    setConsts(juliaC, juliaPlane);

    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    width = x; height = y; 
    // reinitialize with new size
    initPixelBuffer(&pbo, width, height);

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
    initCuda(width, height);

    printf("Press '=' and '-' to change epsilon\n"
           "      'qe,ws,ad,zx' to change c'\n"
           "      'jl,ik,m<' to change plane\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width*2, height*2);
    glutCreateWindow("CUDA volume rendering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initPixelBuffer(&pbo, width, height);

    /* Give c, plane initial values */
    juliaC = make_float4(-0.08f,0,-0.83f,-0.035f);
    juliaPlane = make_float4(0.3f,0.2f,-0.2f,0);
    setConsts(juliaC, juliaPlane);

    glewInit();

    glutMainLoop();

    cleanup(pbo);

    return 0;
}
