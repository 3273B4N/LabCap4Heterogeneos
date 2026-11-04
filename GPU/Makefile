NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo -arch=sm_53
TARGET := softmax_cuda
SRC := softmax_cuda.cu
N ?= 100
REPS ?= 1

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) --precision fp32 --n $(N) --reps $(REPS) --measure kernel

clean:
	rm -f $(TARGET)
