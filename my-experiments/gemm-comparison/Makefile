METAL    := xcrun -sdk macosx metal
METALLIB := xcrun -sdk macosx metallib
CXX      := clang++

METAL_SDK  := $(shell xcrun --sdk macosx --show-sdk-path)

METALFLAGS := \
	-std=metal4.0 \
	-O2 \
	-iframework $(METAL_SDK)/System/Library/Frameworks

CXXFLAGS := \
	-std=c++17 \
	-O2 \
	-arch arm64 \
	-I.
#   -fobjc-arc \
#   -framework Metal \
#   -framework Foundation

CPU_SRCS := src/main.cpp src/cpu/cpu_gemm.cpp
CPU_OBJS := $(CPU_SRCS:%.cpp=build/%.o)
# METAL_SRCS_MM := src/metal/metal_bridge.mm src/metal/naive_gemm.mm src/metal/tiled_gemm.mm
# METAL_OBJS    := $(METAL_SRCS_MM:%.mm=build/%.o)
# METAL_SHADERS := shaders/naive_gemm.metal shaders/tiled_gemm.metal
# METALLIB_OUT  := build/kernels.metallib

TARGET := build/gemm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(CPU_OBJS)  # add $(METAL_OBJS) and $(METALLIB_OUT) when Metal is ready
	$(CXX) $(CXXFLAGS) -o $@ $^

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build/%.o: %.mm
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CXXFLAGS) -fobjc-arc -framework Metal -framework Foundation -c $< -o $@

# $(METALLIB_OUT): $(METAL_SHADERS:%.metal=build/%.air)
# 	$(METALLIB) -o $@ $^

# build/%.air: %.metal
# 	@mkdir -p $(dir $@)
# 	$(METAL) $(METALFLAGS) -c $< -o $@

clean:
	rm -rf build/

