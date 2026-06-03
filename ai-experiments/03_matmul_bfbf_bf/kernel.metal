// 03_matmul_bfbf_bf – bfloat × bfloat → bfloat  (NN, execution_simdgroups<4>)
// bfloat16 has the same exponent range as float32 but only 7 mantissa bits.
// Relevant for ML workloads that trade precision for bandwidth.
//
// Dispatch: MTLSizeMake((N+31)/32, (M+63)/64, 1)  x  128 threads per TG.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_bfbf_bf(
    device bfloat*  A_ptr [[buffer(0)]],
    device bfloat*  B_ptr [[buffer(1)]],
    device bfloat*  C_ptr [[buffer(2)]],
    constant uint&  M     [[buffer(3)]],
    constant uint&  N     [[buffer(4)]],
    constant uint&  K     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor(A_ptr, dextents<int,2>{(int)K,(int)M}, array<int,2>{1,(int)K});
    auto B = tensor(B_ptr, dextents<int,2>{(int)N,(int)K}, array<int,2>{1,(int)N});
    auto C = tensor(C_ptr, dextents<int,2>{(int)N,(int)M}, array<int,2>{1,(int)N});

    constexpr auto desc = matmul2d_descriptor(64, 32,
                              static_cast<int>(metal::dynamic_extent));
    matmul2d<desc, metal::execution_simdgroups<4>> op;

    auto mA = A.slice(0,                (int)tgid.y * 64);
    auto mB = B.slice((int)tgid.x * 32, 0);
    auto mC = C.slice((int)tgid.x * 32, (int)tgid.y * 64);

    op.run(mA, mB, mC);
}
