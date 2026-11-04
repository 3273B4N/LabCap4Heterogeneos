NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14 -lineinfo -arch=sm_53
TARGET := softmax_cuda
SRC := softmax_cuda.cu
ROWS ?= 128
COLS ?= 1024

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) $(ROWS) $(COLS)

clean:
	rm -f $(TARGET)
