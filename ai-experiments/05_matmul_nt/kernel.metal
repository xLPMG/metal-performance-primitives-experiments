// 05_matmul_nt – half × half → float, B transposed (NT layout)
// transpose_right=true means B is stored as K×N (row-major) but is treated
// as if it were transposed, i.e. the logical operation is still A[M×K] @ B^T[K×N].
// This is the common weight-matrix layout in inference (weights are stored
// transposed so each output neuron's weights are contiguous in memory).
//
// MPP tensor convention for NT:
//   A extents: {K, M}, strides: {1, K}   (unchanged from NN)
//   B extents: {K, N}, strides: {1, K}   (note: N is now the slow dim)
//   C extents: {N, M}, strides: {1, N}
//
// Dispatch: MTLSizeMake((N+31)/32, (M+63)/64, 1)  x  128 threads per TG.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_nt(
    device half*   A_ptr [[buffer(0)]],
    device half*   B_ptr [[buffer(1)]],   // stored K×N, treated as transposed
    device float*  C_ptr [[buffer(2)]],
    constant uint& M     [[buffer(3)]],
    constant uint& N     [[buffer(4)]],
    constant uint& K     [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor(A_ptr, dextents<int,2>{(int)K,(int)M}, array<int,2>{1,(int)K});
    // B stored as K rows × N cols, so extents {K,N}, strides {1,K}.
    auto B = tensor(B_ptr, dextents<int,2>{(int)K,(int)N}, array<int,2>{1,(int)K});
    auto C = tensor(C_ptr, dextents<int,2>{(int)N,(int)M}, array<int,2>{1,(int)N});

    // transpose_left=false, transpose_right=true  →  NT
    constexpr auto desc = matmul2d_descriptor(64, 32,
                              static_cast<int>(metal::dynamic_extent),
                              /*transpose_left=*/false,
                              /*transpose_right=*/true);
    matmul2d<desc, metal::execution_simdgroups<4>> op;

    auto mA = A.slice(0,                (int)tgid.y * 64);
    auto mB = B.slice(0,                (int)tgid.x * 32);  // offset N dim (dim 1)
    auto mC = C.slice((int)tgid.x * 32, (int)tgid.y * 64);

    op.run(mA, mB, mC);
}
