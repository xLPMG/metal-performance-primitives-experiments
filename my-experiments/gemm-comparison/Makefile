METAL    := xcrun -sdk macosx metal
METALLIB := xcrun -sdk macosx metallib
CXX      := clang++

METAL_SDK := $(shell xcrun --sdk macosx --show-sdk-path)

METALFLAGS := \
	-std=metal4.0 \
	-O2 \
	-iframework $(METAL_SDK)/System/Library/Frameworks

# Compile flags (no linker flags here)
CXXFLAGS := \
	-std=c++17 \
	-O2 \
	-arch arm64 \
	-I.

OBJCFLAGS := -fobjc-arc

# Link flags only
LDFLAGS := \
	-framework Metal \
	-framework Foundation \
	-framework MetalPerformanceShaders

CPU_SRCS      := src/main.cpp src/cpu/cpu_gemm.cpp
CPU_OBJS      := $(CPU_SRCS:%.cpp=build/%.o)
METAL_SRCS    := src/metal/metal_bridge.mm
METAL_OBJS    := $(METAL_SRCS:%.mm=build/%.o)
METAL_SHADERS := shaders/gemm_naive.metal shaders/gemm_tiled.metal
METALLIB_OUT  := build/kernels.metallib

TARGET := build/gemm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(CPU_OBJS) $(METAL_OBJS) $(METALLIB_OUT)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(CPU_OBJS) $(METAL_OBJS)

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -c $< -o $@

$(METALLIB_OUT): $(METAL_SHADERS:%.metal=build/%.air)
	$(METALLIB) -o $@ $^

build/%.air: %.metal
	@mkdir -p $(dir $@)
	$(METAL) $(METALFLAGS) -c $< -o $@

clean:
	rm -rf build/
