METAL     := xcrun -sdk macosx metal
METALLIB  := xcrun -sdk macosx metallib
CXX       := clang++
METAL_SDK := $(shell xcrun --sdk macosx --show-sdk-path)

METALFLAGS := \
	-std=metal4.0 \
	-O2 \
	-iframework $(METAL_SDK)/System/Library/Frameworks

CXXFLAGS := \
	-std=c++17 \
	-O2 \
	-arch arm64

LDFLAGS := \
	-fobjc-arc \
	-framework Metal \
	-framework Foundation

TILES    := 32x16 32x32 64x32 64x64 128x32 128x64 128x128
SIMDGRPS := $(shell seq 1 32)

COMBOS    := $(foreach t,$(TILES),$(foreach s,$(SIMDGRPS),$(t)_sg$(s)))
METALLIBS := $(COMBOS:%=build/%.metallib)
TARGET    := build/bench

.PHONY: all generate clean

all: generate $(TARGET) $(METALLIBS)

generate:
	python3 gen.py

$(TARGET): src/main.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

build/%.metallib: shaders/mpp_%.metal
	@mkdir -p $(dir $@)
	@if $(METAL) $(METALFLAGS) -c $< -o $(@:.metallib=.air) 2>/dev/null && \
	    $(METALLIB) -o $@ $(@:.metallib=.air) 2>/dev/null; then \
	    true; \
	else \
	    echo "REJECTED: $<"; \
	fi

clean:
	rm -rf build/ shaders/mpp_*.metal
