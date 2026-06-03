// 09_matmul_relu_fused – half × half → float, ReLU applied in-register
//
// Demonstrates the cooperative tensor path:
//   matmul2d::run() leaves the output tile in registers (never touches device
//   memory for C), then each thread applies ReLU to its elements before a
//   single cT.store() writes the final result.
//
// This is the core fusion pattern relevant to TEIR lowering:
//   %C = matmul(%A, %B)   ->  cooperative_tensor (in register)
//   %D = relu(%C)          ->  in-register, per-element, zero extra bandwidth
//   store %D               ->  one write to device memory
//
// Dispatch: MTLSizeMake((N+31)/32, (M+63)/64, 1)  x  128 threads per TG.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_relu_fused(
    device half*   A_ptr [[buffer(0)]],
    device half*   B_ptr [[buffer(1)]],
    device float*  C_ptr [[buffer(2)]],   // output: relu(A @ B), float32
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // ── Build tensor views ──────────────────────────────────────────────────
    auto A = tensor(A_ptr, dextents<int,2>{(int)K,(int)M}, array<int,2>{1,(int)K});
    auto B = tensor(B_ptr, dextents<int,2>{(int)N,(int)K}, array<int,2>{1,(int)N});
    auto C = tensor(C_ptr, dextents<int,2>{(int)N,(int)M}, array<int,2>{1,(int)N});

    // ── Tile slices for this threadgroup ────────────────────────────────────
    auto mA = A.slice(0,                (int)tgid.y * 64);
    auto mB = B.slice((int)tgid.x * 32, 0);
    auto mC = C.slice((int)tgid.x * 32, (int)tgid.y * 64);

    // ── Create the op ───────────────────────────────────────────────────────
    constexpr auto desc = matmul2d_descriptor(64, 32,
                              static_cast<int>(metal::dynamic_extent));
    matmul2d<desc, metal::execution_simdgroups<4>> op;

    // ── Allocate a cooperative_tensor for the destination (float32) ─────────
    // This keeps the tile in registers — no device-memory write yet.
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();

    // Zero-initialise every slot in this thread's fragment.
    // [[clang::unroll]] avoids dynamic indexing into register-resident data.
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT[i] = 0.f;

    // ── Run matmul into cooperative_tensor ──────────────────────────────────
    // Result is distributed in registers across all 4 simdgroups.
    // No memory traffic for C at this point.
    op.run(mA, mB, cT);

    // ── Apply ReLU in-register ───────────────────────────────────────────────
    // get_multidimensional_index(i) returns the (col, row) coordinate of
    // element i within the local 32×64 tile, using the opaque chip layout.
    // We don't actually need the coordinate for ReLU, but it's shown here
    // because it's the key API for position-dependent fusions (bias, masking…).
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        // get_multidimensional_index(i) returns (col, row) within the local tile.
        // Useful for position-dependent fusions like bias add or causal masking:
        //   auto idx = cT.get_multidimensional_index(i);
        //   int row = idx[1], col = idx[0];

        cT[i] = max(cT[i], 0.f);   // ReLU: clamp negatives to zero
    }

    // ── Single write to device memory ──────────────────────────────────────
    // This is the only time C is touched — half the bandwidth of a naive
    // two-kernel approach (write C, read C, write relu(C)).
    cT.store(mC);
}
