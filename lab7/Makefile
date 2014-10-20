# CC = /cs/courses/cs179/openmpi-1.6.4/bin/mpic++
CC = /usr/local/bin/mpic++

LD_FLAGS = -lrt

#CUDA_PATH       ?= /cs/courses/cs179/public_html/cuda-5.0
CUDA_PATH       ?= /Developer/NVIDIA/CUDA-5.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
#CUDA_LIB_PATH   ?= /usr/lib64
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30

# Common binaries
#NVCC            ?= $(CUDA_BIN_PATH)/nvcc
NVCC            ?= $(CUDA_PATH)/bin/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcufft
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -lcufft
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = MPI_CUDA_wave

all: $(TARGETS)

MPI_CUDA_wave: MPI_CUDA_wave.cc Cuda1DFDWave_cuda.o
	$(CC) $< -o $@ Cuda1DFDWave_cuda.o -O3 $(LDFLAGS) -Wall -std=c++0x -I$(CUDA_INC_PATH)

Cuda1DFDWave_cuda.o: Cuda1DFDWave_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
