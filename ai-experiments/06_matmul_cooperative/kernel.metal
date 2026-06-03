// 06_matmul_cooperative – half × half → float with bias add via cooperative_tensor
// Demonstrates the full cooperative tensor workflow:
//   1. matmul result kept in registers (cooperative_tensor)
//   2. bias vector loaded into a second cooperative_tensor
//   3. bias added in-register using get_multidimensional_index for column lookup
//   4. single cT.store() to device memory
//
// Dispatch: MTLSizeMake((N+31)/32, (M+63)/64, 1)  x  128 threads per TG.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_bias_cooperative(
    device half*   A_ptr    [[buffer(0)]],
    device half*   B_ptr    [[buffer(1)]],
    device float*  C_ptr    [[buffer(2)]],
    device float*  bias_ptr [[buffer(3)]],   // bias[N]: one value per output column
    constant uint& M        [[buffer(4)]],
    constant uint& N        [[buffer(5)]],
    constant uint& K        [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor(A_ptr, dextents<int,2>{(int)K,(int)M}, array<int,2>{1,(int)K});
    auto B = tensor(B_ptr, dextents<int,2>{(int)N,(int)K}, array<int,2>{1,(int)N});
    auto C = tensor(C_ptr, dextents<int,2>{(int)N,(int)M}, array<int,2>{1,(int)N});

    auto mA = A.slice(0,                (int)tgid.y * 64);
    auto mB = B.slice((int)tgid.x * 32, 0);
    auto mC = C.slice((int)tgid.x * 32, (int)tgid.y * 64);

    constexpr auto desc = matmul2d_descriptor(64, 32,
                              static_cast<int>(metal::dynamic_extent));
    matmul2d<desc, metal::execution_simdgroups<4>> op;

    // ── Destination cooperative_tensor (stays in registers) ─────────────────
    auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();

    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        cT[i] = 0.f;

    op.run(mA, mB, cT);

    // ── Fused bias add in-register using get_multidimensional_index ──────────
    // idx[0] = column within the local 32-wide N tile (0..31)
    // Absolute column in N = tgid.x * 32 + idx[0]
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        auto idx = cT.get_multidimensional_index(i);
        int abs_col = (int)tgid.x * 32 + (int)idx[0];
        cT[i] += bias_ptr[abs_col];
    }

    cT.store(mC);
}
