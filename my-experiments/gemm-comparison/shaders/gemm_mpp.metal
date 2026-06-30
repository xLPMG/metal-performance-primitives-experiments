#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Targeted using declarations for learning
using metal::array;
using metal::dextents;
using metal::tensor;
using mpp::tensor_ops::matmul2d_descriptor;
using mpp::tensor_ops::matmul2d;

kernel void gemm_mpp(   device half* A [[buffer(0)]],
                        device half* B [[buffer(1)]],
                        device half* C [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& N [[buffer(4)]],
                        constant uint& K [[buffer(5)]],
                        uint2 tgid [[threadgroup_position_in_grid]])
{
    // Wrap raw device pointers into MPP tensor views.
    // Dimensions are listed as {inner, outer}, i.e. {col_dim, row_dim}.
    // Strides are {1, leading_dim} for row-major layout:
    //   A[m,k] = A_ptr[m*K + k]  ->  shape{K,M}, strides{1,K}
    //   B[k,n] = B_ptr[k*N + n]  ->  shape{N,K}, strides{1,N}
    //   C[m,n] = C_ptr[m*N + n]  ->  shape{N,M}, strides{1,N}
    // ── Tile configuration ──────────────────────────────────────────────────
    // When changing these, update metal_bridge.mm to match:
    //   grid = MTLSizeMake((N + N_TILE-1)/N_TILE, (M + M_TILE-1)/M_TILE, 1)
    //   tpg  = MTLSizeMake(32, SG, 1)
    constexpr int M_TILE = 64;
    constexpr int N_TILE = 32;
    constexpr int SG     = 4;  // simdgroups; threads per threadgroup = SG * 32

    auto tensorA = tensor(A, dextents<int,2>{K, M}, array<int,2>{1, (int)K});
    auto tensorB = tensor(B, dextents<int,2>{N, K}, array<int,2>{1, (int)N});
    auto tensorC = tensor(C, dextents<int,2>{N, M}, array<int,2>{1, (int)N});

    auto mA = tensorA.slice(0,               tgid.y * M_TILE);
    auto mB = tensorB.slice(tgid.x * N_TILE, 0);
    auto mC = tensorC.slice(tgid.x * N_TILE, tgid.y * M_TILE);

    constexpr auto desc = matmul2d_descriptor(M_TILE, N_TILE, static_cast<int>(metal::dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SG>> op;

    op.run(mA, mB, mC);
}