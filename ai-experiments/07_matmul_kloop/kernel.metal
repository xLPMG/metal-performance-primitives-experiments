// 07_matmul_kloop – half × half → float, host-controlled K-tiling
// Instead of letting MPP internally loop over K with dynamic_extent, the host
// kernel explicitly tiles K in chunks of TILEK=32, accumulating into a
// cooperative_tensor in registers. The intermediate results never touch
// device memory; only a single cT.store() at the end writes to C.
//
// Dispatch: MTLSizeMake((N+31)/32, (M+63)/64, 1)  x  128 threads per TG.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant int TILEK = 32;

kernel void matmul_kloop(
    device half*   A_ptr [[buffer(0)]],
    device half*   B_ptr [[buffer(1)]],
    device float*  C_ptr [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor(A_ptr, dextents<int,2>{(int)K,(int)M}, array<int,2>{1,(int)K});
    auto B = tensor(B_ptr, dextents<int,2>{(int)N,(int)K}, array<int,2>{1,(int)N});
    auto C = tensor(C_ptr, dextents<int,2>{(int)N,(int)M}, array<int,2>{1,(int)N});

    auto mC = C.slice((int)tgid.x * 32, (int)tgid.y * 64);

    // Static k tile: descriptor k=TILEK is a compile-time constant.
    // Default mode (multiply): first call overwrites cT; subsequent calls
    // use multiply_accumulate mode so partial sums pile up in registers.
    constexpr auto desc_first = matmul2d_descriptor(
        64, 32, TILEK,
        /*transpose_left=*/false, /*transpose_right=*/false,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto desc_acc = matmul2d_descriptor(
        64, 32, TILEK,
        /*transpose_left=*/false, /*transpose_right=*/false,
        /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);

    matmul2d<desc_first, metal::execution_simdgroups<4>> op_first;
    matmul2d<desc_acc,   metal::execution_simdgroups<4>> op_acc;

    // First K tile — get the destination cooperative_tensor via op_first
    auto cT = op_first.get_destination_cooperative_tensor<
                   decltype(A.slice(0,0)), decltype(B.slice(0,0)), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0.f;

    // Loop: use op_first for k=0, op_acc for remaining tiles
    for (int k = 0; k < (int)K; k += TILEK) {
        auto tA = A.slice(k,                (int)tgid.y * 64);
        auto tB = B.slice((int)tgid.x * 32, k);
        if (k == 0) op_first.run(tA, tB, cT);
        else        op_acc.run(tA, tB, cT);
    }

    cT.store(mC);
}
