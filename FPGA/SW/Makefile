BENCH := bench_fp32 bench_fp16

CFLAGS := -std=c++17 -O2 -I/usr/include/xrt
LDFLAGS := -L/opt/xilinx/xrt/lib -lxrt_coreutil -lpthread

.PHONY: all clean

all: $(BENCH)

bench_fp32: bench_softmax.cpp
	$(CXX) $(CFLAGS) $< -o $@ $(LDFLAGS)

bench_fp16: bench_softmax.cpp
	$(CXX) $(CFLAGS) -DFP16 $< -o $@ $(LDFLAGS)

clean:
	$(RM) $(BENCH)
