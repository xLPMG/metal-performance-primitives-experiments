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
	-fobjc-arc \
	-framework Metal \
	-framework Foundation \
	-I.

EXPS := \
	01_matmul_hh_h \
	02_matmul_hh_f \
	03_matmul_bfbf_bf \
	04_matmul_i8i8_i32 \
	05_matmul_nt \
	06_matmul_cooperative \
	07_matmul_kloop \
	08_conv2d_basic \
	09_matmul_relu_fused

.PHONY: all clean run \
	run_01_matmul_hh_h run_02_matmul_hh_f run_03_matmul_bfbf_bf \
	run_04_matmul_i8i8_i32 run_05_matmul_nt run_06_matmul_cooperative \
	run_07_matmul_kloop run_08_conv2d_basic run_09_matmul_relu_fused

all: \
	build/01_matmul_hh_h/bench \
	build/02_matmul_hh_f/bench \
	build/03_matmul_bfbf_bf/bench \
	build/04_matmul_i8i8_i32/bench \
	build/05_matmul_nt/bench \
	build/06_matmul_cooperative/bench \
	build/07_matmul_kloop/bench \
	build/08_conv2d_basic/bench \
	build/09_matmul_relu_fused/bench

# ── 01 ──────────────────────────────────────────────────────────────────────
build/01_matmul_hh_h:
	mkdir -p $@
build/01_matmul_hh_h/kernel.air: 01_matmul_hh_h/kernel.metal | build/01_matmul_hh_h
	$(METAL) $(METALFLAGS) -c $< -o $@
build/01_matmul_hh_h/kernel.metallib: build/01_matmul_hh_h/kernel.air
	$(METALLIB) $< -o $@
build/01_matmul_hh_h/bench: 01_matmul_hh_h/main.mm common/bench.h common/half_utils.h build/01_matmul_hh_h/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/01_matmul_hh_h/kernel.metallib"' $< -o $@
run_01_matmul_hh_h: build/01_matmul_hh_h/bench
	./build/01_matmul_hh_h/bench

# ── 02 ──────────────────────────────────────────────────────────────────────
build/02_matmul_hh_f:
	mkdir -p $@
build/02_matmul_hh_f/kernel.air: 02_matmul_hh_f/kernel.metal | build/02_matmul_hh_f
	$(METAL) $(METALFLAGS) -c $< -o $@
build/02_matmul_hh_f/kernel.metallib: build/02_matmul_hh_f/kernel.air
	$(METALLIB) $< -o $@
build/02_matmul_hh_f/bench: 02_matmul_hh_f/main.mm common/bench.h common/half_utils.h build/02_matmul_hh_f/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/02_matmul_hh_f/kernel.metallib"' $< -o $@
run_02_matmul_hh_f: build/02_matmul_hh_f/bench
	./build/02_matmul_hh_f/bench

# ── 03 ──────────────────────────────────────────────────────────────────────
build/03_matmul_bfbf_bf:
	mkdir -p $@
build/03_matmul_bfbf_bf/kernel.air: 03_matmul_bfbf_bf/kernel.metal | build/03_matmul_bfbf_bf
	$(METAL) $(METALFLAGS) -c $< -o $@
build/03_matmul_bfbf_bf/kernel.metallib: build/03_matmul_bfbf_bf/kernel.air
	$(METALLIB) $< -o $@
build/03_matmul_bfbf_bf/bench: 03_matmul_bfbf_bf/main.mm common/bench.h common/half_utils.h build/03_matmul_bfbf_bf/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/03_matmul_bfbf_bf/kernel.metallib"' $< -o $@
run_03_matmul_bfbf_bf: build/03_matmul_bfbf_bf/bench
	./build/03_matmul_bfbf_bf/bench

# ── 04 ──────────────────────────────────────────────────────────────────────
build/04_matmul_i8i8_i32:
	mkdir -p $@
build/04_matmul_i8i8_i32/kernel.air: 04_matmul_i8i8_i32/kernel.metal | build/04_matmul_i8i8_i32
	$(METAL) $(METALFLAGS) -c $< -o $@
build/04_matmul_i8i8_i32/kernel.metallib: build/04_matmul_i8i8_i32/kernel.air
	$(METALLIB) $< -o $@
build/04_matmul_i8i8_i32/bench: 04_matmul_i8i8_i32/main.mm common/bench.h build/04_matmul_i8i8_i32/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/04_matmul_i8i8_i32/kernel.metallib"' $< -o $@
run_04_matmul_i8i8_i32: build/04_matmul_i8i8_i32/bench
	./build/04_matmul_i8i8_i32/bench

# ── 05 ──────────────────────────────────────────────────────────────────────
build/05_matmul_nt:
	mkdir -p $@
build/05_matmul_nt/kernel.air: 05_matmul_nt/kernel.metal | build/05_matmul_nt
	$(METAL) $(METALFLAGS) -c $< -o $@
build/05_matmul_nt/kernel.metallib: build/05_matmul_nt/kernel.air
	$(METALLIB) $< -o $@
build/05_matmul_nt/bench: 05_matmul_nt/main.mm common/bench.h common/half_utils.h build/05_matmul_nt/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/05_matmul_nt/kernel.metallib"' $< -o $@
run_05_matmul_nt: build/05_matmul_nt/bench
	./build/05_matmul_nt/bench

# ── 06 ──────────────────────────────────────────────────────────────────────
build/06_matmul_cooperative:
	mkdir -p $@
build/06_matmul_cooperative/kernel.air: 06_matmul_cooperative/kernel.metal | build/06_matmul_cooperative
	$(METAL) $(METALFLAGS) -c $< -o $@
build/06_matmul_cooperative/kernel.metallib: build/06_matmul_cooperative/kernel.air
	$(METALLIB) $< -o $@
build/06_matmul_cooperative/bench: 06_matmul_cooperative/main.mm common/bench.h common/half_utils.h build/06_matmul_cooperative/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/06_matmul_cooperative/kernel.metallib"' $< -o $@
run_06_matmul_cooperative: build/06_matmul_cooperative/bench
	./build/06_matmul_cooperative/bench

# ── 07 ──────────────────────────────────────────────────────────────────────
build/07_matmul_kloop:
	mkdir -p $@
build/07_matmul_kloop/kernel.air: 07_matmul_kloop/kernel.metal | build/07_matmul_kloop
	$(METAL) $(METALFLAGS) -c $< -o $@
build/07_matmul_kloop/kernel.metallib: build/07_matmul_kloop/kernel.air
	$(METALLIB) $< -o $@
build/07_matmul_kloop/bench: 07_matmul_kloop/main.mm common/bench.h common/half_utils.h build/07_matmul_kloop/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/07_matmul_kloop/kernel.metallib"' $< -o $@
run_07_matmul_kloop: build/07_matmul_kloop/bench
	./build/07_matmul_kloop/bench

# ── 08 ──────────────────────────────────────────────────────────────────────
build/08_conv2d_basic:
	mkdir -p $@
build/08_conv2d_basic/kernel.air: 08_conv2d_basic/kernel.metal | build/08_conv2d_basic
	$(METAL) $(METALFLAGS) -c $< -o $@
build/08_conv2d_basic/kernel.metallib: build/08_conv2d_basic/kernel.air
	$(METALLIB) $< -o $@
build/08_conv2d_basic/bench: 08_conv2d_basic/main.mm common/bench.h common/half_utils.h build/08_conv2d_basic/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/08_conv2d_basic/kernel.metallib"' $< -o $@
run_08_conv2d_basic: build/08_conv2d_basic/bench
	./build/08_conv2d_basic/bench

run: all
	@for exp in $(EXPS); do \
		echo ""; \
		echo "=== $$exp ==="; \
		./build/$$exp/bench; \
	done

# ── 09 ──────────────────────────────────────────────────────────────────────
build/09_matmul_relu_fused:
	mkdir -p $@
build/09_matmul_relu_fused/kernel.air: 09_matmul_relu_fused/kernel.metal | build/09_matmul_relu_fused
	$(METAL) $(METALFLAGS) -c $< -o $@
build/09_matmul_relu_fused/kernel.metallib: build/09_matmul_relu_fused/kernel.air
	$(METALLIB) $< -o $@
build/09_matmul_relu_fused/bench: 09_matmul_relu_fused/main.mm common/bench.h common/half_utils.h build/09_matmul_relu_fused/kernel.metallib
	$(CXX) $(CXXFLAGS) -DMETALLIB_PATH='"build/09_matmul_relu_fused/kernel.metallib"' $< -o $@
run_09_matmul_relu_fused: build/09_matmul_relu_fused/bench
	./build/09_matmul_relu_fused/bench

clean:
	rm -rf build
