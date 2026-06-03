// Simple Metal Performance Primitives – matmul POC
// Computes C = A @ B
//   A: M x K  (half, row-major)
//   B: K x N  (half, row-major)
//   C: M x N  (half, row-major)
//
// execution_simdgroups<4> supports half x half -> half.
// (half x half -> float is only valid with execution_thread scope.)
//
// Tile: m=64, n=32, with 4 simdgroups per threadgroup.
// The MPP tensor convention for NN (no-transpose) matmul:
//   A tensor extents: [K, M], strides: [1, K]
//   B tensor extents: [N, K], strides: [1, N]
//   C tensor extents: [N, M], strides: [1, N]
//
// Dispatch from host: MTLSizeMake((N+31)/32, (M+63)/64, 1)

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_mpp(
    device half*  A_ptr [[buffer(0)]],
    device half*  B_ptr [[buffer(1)]],
    device half*  C_ptr [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor(A_ptr,
                    dextents<int, 2>{(int)K, (int)M},
                    array<int, 2>{1, (int)K});

    auto B = tensor(B_ptr,
                    dextents<int, 2>{(int)N, (int)K},
                    array<int, 2>{1, (int)N});

    auto C = tensor(C_ptr,
                    dextents<int, 2>{(int)N, (int)M},
                    array<int, 2>{1, (int)N});

    // Tile: 64 rows of M, 32 cols of N, dynamic K.
    // execution_simdgroups<4>: all 4 simdgroups in this threadgroup cooperate.
    constexpr auto desc = matmul2d_descriptor(64, 32,
                                              static_cast<int>(metal::dynamic_extent));
    matmul2d<desc, metal::execution_simdgroups<4>> op;

    auto mA = A.slice(0,           tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    // C must be zero-initialised by the host before dispatch.
    op.run(mA, mB, mC);
}
