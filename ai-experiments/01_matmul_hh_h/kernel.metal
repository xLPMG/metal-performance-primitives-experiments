// 01_matmul_hh_h – half × half → half  (NN, execution_simdgroups<4>)
// Tile: m=64, n=32, k=dynamic.
// Dispatch: MTLSizeMake((N+31)/32, (M+63)/64, 1)  x  128 threads per TG.
//
// MPP tensor convention (fastest dimension first, row-major matrices):
//   A [M×K row-major]: extents {K,M}, strides {1,K}
//   B [K×N row-major]: extents {N,K}, strides {1,N}
//   C [M×N row-major]: extents {N,M}, strides {1,N}

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_hh_h(
    device half*   A_ptr [[buffer(0)]],
    device half*   B_ptr [[buffer(1)]],
    device half*   C_ptr [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor(A_ptr, dextents<int,2>{(int)K,(int)M}, array<int,2>{1,(int)K});
    auto B = tensor(B_ptr, dextents<int,2>{(int)N,(int)K}, array<int,2>{1,(int)N});
    auto C = tensor(C_ptr, dextents<int,2>{(int)N,(int)M}, array<int,2>{1,(int)N});

    constexpr auto desc = matmul2d_descriptor(64, 32,
                              static_cast<int>(metal::dynamic_extent));
    matmul2d<desc, metal::execution_simdgroups<4>> op;

    auto mA = A.slice(0,             (int)tgid.y * 64);
    auto mB = B.slice((int)tgid.x * 32, 0);
    auto mC = C.slice((int)tgid.x * 32, (int)tgid.y * 64);

    op.run(mA, mB, mC);
}
