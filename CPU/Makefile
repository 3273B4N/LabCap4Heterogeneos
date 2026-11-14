CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native
TARGET := softmax_cpu
SRC := softmax_cpu.cpp
N ?= 100
REPS ?= 10000

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ -lm

run: $(TARGET)
	./$(TARGET) --precision fp32 --n $(N) --reps $(REPS) --measure function

clean:
	rm -f $(TARGET)
