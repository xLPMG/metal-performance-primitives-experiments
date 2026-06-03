CXX        := clang++
METAL      := xcrun -sdk macosx metal
METALLIB   := xcrun -sdk macosx metallib

METAL_SDK  := $(shell xcrun --sdk macosx --show-sdk-path)
MPP_HEADERS := $(METAL_SDK)/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers

# Host compiler flags
CXXFLAGS := \
    -std=c++17 \
    -O2 \
    -arch arm64 \
    -fobjc-arc \
    -framework Metal \
    -framework Foundation

# Metal shader compiler flags
# metal4.0 is required to enable __HAVE_TENSOR__ and the MPP tensor ops API.
METALFLAGS := \
    -std=metal4.0 \
    -O2 \
    -iframework $(METAL_SDK)/System/Library/Frameworks

SRC_DIR   := src
BUILD_DIR := build

METAL_SRC   := $(SRC_DIR)/matmul.metal
AIR_OUT     := $(BUILD_DIR)/matmul.air
METALLIB_OUT := $(BUILD_DIR)/kernels.metallib

HOST_SRC := $(SRC_DIR)/main.mm
BINARY   := $(BUILD_DIR)/mpp_poc

.PHONY: all run clean

all: $(METALLIB_OUT) $(BINARY)

$(BUILD_DIR):
	mkdir -p $@

# Compile .metal → .air
$(AIR_OUT): $(METAL_SRC) | $(BUILD_DIR)
	$(METAL) $(METALFLAGS) -c $< -o $@

# Link .air → .metallib
$(METALLIB_OUT): $(AIR_OUT)
	$(METALLIB) $< -o $@

# Compile host code
$(BINARY): $(HOST_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run: all
	$(BINARY)

clean:
	rm -rf $(BUILD_DIR)
