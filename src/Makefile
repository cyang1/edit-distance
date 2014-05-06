
EXECUTABLE := benchmark

CU_FILES   := cuda_parallel_simple.cu

CU_DEPS    :=

CC_FILES   := benchmark.cpp simple_parallel.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/cuda/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 -arch=sm_20


OBJS=$(OBJDIR)/benchmark.o $(OBJDIR)/implementations/simple_parallel/simple_parallel.o \
     $(OBJDIR)/implementations/cuda_parallel/cuda_parallel_simple.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/ $(OBJDIR)/implementations/simple_parallel \
		    $(OBJDIR)/implementations/cuda_parallel

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
