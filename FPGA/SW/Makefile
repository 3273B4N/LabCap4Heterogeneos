# List of source files and corresponding executables
SOURCES := SoftMax.cpp
EXECUTABLES := SoftMax

CFLAGS := -I/usr/include -Wall -std=c++17 -O0 -g -I.
LDFLAGS := -L/opt/xilinx/xrt/lib -pthread -lxrt_core -lxrt_coreutil

.PHONY: all clean

# Default target to build all executables
all: $(EXECUTABLES)

# Rule to build each executable from its corresponding source file
$(EXECUTABLES): %: %.cpp
	$(CXX) -o $@ $< $(CFLAGS) $(LDFLAGS)

# Clean target to remove all executables
clean:
	$(RM) $(EXECUTABLES)

