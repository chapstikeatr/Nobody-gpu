NVCC=nvcc
ARCH=-arch=sm_61
CXXFLAGS=-O3 -std=c++17

SRC=nbody_gpu.cpp
EXE=nbody_gpu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(ARCH) $(CXXFLAGS) -x cu $(SRC) -o $(EXE)

clean:
	rm -f $(EXE)
