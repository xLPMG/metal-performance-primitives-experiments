# MPP Experiments

Minimal benchmarks for each Metal Performance Primitives (MPP) primitive.

## Building

Requires macOS with Xcode Command Line Tools. Uses BSD make (ships with macOS — **not** GNU make).

```
make          # build all experiments
make run      # build + run all
make run_01_matmul_hh_h   # run a single experiment
```

Build artefacts land in `build/`.

## Common infrastructure

| File | Purpose |
|---|---|
| `common/bench.h` | GPU timing harness. `bench(n_warmup, n_iter, fn)` returns min/avg/max in ms and prints TFLOPS. |
| `common/half_utils.h` | CPU-side `float16`/`bfloat16` conversion (`f32_to_f16`, `f16_to_f32`, `f32_to_bf16`, `bf16_to_f32`). |

## Experiments

### 01 · `matmul_hh_h` — half × half → half (NN)
Baseline `matmul2d` using the standard NN tensor layout. Both inputs and the output are `half`.
Tile: 64×32, 4 simdgroups, dynamic K.

### 02 · `matmul_hh_f` — half × half → float (NN)
Same as 01 but accumulates into a `float` output buffer. Demonstrates that the accumulator
precision and the output type can differ from the input types.

### 03 · `matmul_bfbf_bf` — bfloat × bfloat → bfloat (NN)
Substitutes `bfloat` throughout. Exercises the bfloat16 path of the MPP descriptor with an
otherwise identical kernel structure to 01.

### 04 · `matmul_i8i8_i32` — int8 × int8 → int32 (NN)
Integer GEMM: `int8_t` inputs, `int32_t` accumulator. Demonstrates the integer data-type path
and the larger dynamic range of the int32 output.

### 05 · `matmul_nt` — half × half → float (NT)
Non-transposed A, transposed B (`transpose_right = true`). B is stored N×K row-major
(each row = the weight vector of one output neuron), which is the common layout for
inference weight matrices. The CPU reference indexes B as `B[n*K + k]` to match.

### 06 · `matmul_cooperative` — matmul + bias fused (cooperative tensor)
Keeps the matmul result in registers as a `cooperative_tensor` instead of writing it to
device memory. A per-column bias is added in-register using `get_multidimensional_index(i)`
to map each register element to its column index, then a single `cT.store()` writes C.
This is the canonical pattern for zero-overhead elementwise fusions.

### 07 · `matmul_kloop` — explicit K-tiling via cooperative tensor accumulation
The kernel manually loops over K in tiles of 32. The first tile uses `mode::multiply`
(overwrites the cooperative tensor), subsequent tiles use `mode::multiply_accumulate`
(adds in-register). A single `cT.store()` at the end writes C. Intermediate partial sums
never touch device memory.

### 08 · `conv2d_basic` — 3×3 convolution NHWC/HWIO half → half
A minimal `convolution2d` kernel. Activation is NHWC, weights are HWIO (both required by MPP).
Each threadgroup processes one output spatial position via `op.set_offsets(int2(tgid.x, tgid.y))`.
Problem size: 1×8×8×16 → 1×6×6×32 (no padding, stride 1).

### 09 · `matmul_relu_fused` — matmul + ReLU fused (cooperative tensor)
Matmul result stays in registers as a `cooperative_tensor`. ReLU is applied element-wise
with `cT[i] = max(cT[i], 0.f)` before `cT.store()`. Shows `get_multidimensional_index`
as the hook for position-aware fusions (masking, softmax, etc.) even though ReLU itself
doesn't need positional info.

## Measured results (Apple M3 Pro)

| # | Experiment | Min (ms) | Avg (ms) | TFLOPS |
|---|---|---|---|---|
| 01 | matmul half×half→half | 3.73 | 3.85 | 4.60 |
| 02 | matmul half×half→float | 3.72 | 3.82 | 4.62 |
| 03 | matmul bfloat×bfloat→bfloat | 3.74 | 3.82 | 4.60 |
| 04 | matmul int8×int8→int32 | 4.22 | 4.31 | 4.07 |
| 05 | matmul NT half×half→float | 3.76 | 3.87 | 4.57 |
| 06 | matmul+bias (cooperative) | 3.67 | 3.74 | 4.69 |
| 07 | matmul kloop (tilek=32) | 3.70 | 3.82 | 4.65 |
| 08 | conv2d 3×3 NHWC half→half | 0.029 | 0.052 | 0.01 |
| 09 | matmul+ReLU (cooperative) | 3.66 | 3.77 | 4.69 |

Problem size for matmul benchmarks: M=N=K=2048. Benchmark: 10 warmup + 100 measured iterations.
