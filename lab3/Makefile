################################################################################
### CHANGE THESE LINES TO MATCH YOUR SYSTEM                                  ###
### COMPILER PATH                                                            ###
CC = /usr/bin/g++
### CUDA FOLDER PATH                                                         ###
CUDA_PATH       ?= /cs/courses/cs179/public_html/cuda-5.0
# CUDA code generation flags                                                 ###
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20
# library flags -- on linux, this may look like -lgl -lglut.                 ###
#                  on mac, it would look like -framework OpenGL              ###
#                  -framework GLUT                                           ###
LD_FLAGS = -lGL -lGLU -lglut
# includes for some helper functions -- if this doesn't work, you may not    ###
# have downloaded the CUDA SDK.                                              ###
CC_INCLUDE = -I$(CUDA_PATH)/samples/common/inc
################################################################################

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = particles_interact

all: $(TARGETS)

particles_interact: particles_interact.cc interact_kernel.o
	$(CC) $< -o $@ interact_kernel.o -O3 -I$(CUDA_INC_PATH) $(LDFLAGS) $(LD_FLAGS) -Wall

interact_kernel.o: interact_kernel.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(LD_FLAGS) -I$(CUDA_INC_PATH) $(CC_INCLUDE) -o $@ -c $<


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
