#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using metal::array;
using metal::dextents;
using metal::tensor;
using mpp::tensor_ops::matmul2d_descriptor;
using mpp::tensor_ops::matmul2d;

// Morton decode: extract even bits → col, odd bits → row.
// Maps a linear threadgroup index to a 2D tile coordinate using a Z-curve,
// so adjacent indices in dispatch order share cached data in both dimensions.
// Requires grid_x and grid_y to both be powers of two.
static uint morton_x(uint n) {
    n &= 0x55555555u;
    n = (n | (n >>  1u)) & 0x33333333u;
    n = (n | (n >>  2u)) & 0x0f0f0f0fu;
    n = (n | (n >>  4u)) & 0x00ff00ffu;
    n = (n | (n >>  8u)) & 0x0000ffffu;
    return n;
}
static uint morton_y(uint n) { return morton_x(n >> 1u); }

kernel void gemm_mpp_morton(
    device half*   A [[buffer(0)]],
    device half*   B [[buffer(1)]],
    device half*   C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tgid_linear [[threadgroup_position_in_grid]])
{
    // ── Tile configuration ──────────────────────────────────────────────────
    // When changing these, update metal_bridge.mm to match:
    //   tpg  = MTLSizeMake(SG * 32, 1, 1)
    //   grid = MTLSizeMake(grid_x * grid_y, 1, 1)
    //          where grid_x = (N+N_TILE-1)/N_TILE, grid_y = (M+M_TILE-1)/M_TILE
    constexpr int M_TILE = 64;
    constexpr int N_TILE = 32;
    constexpr int SG     = 4;  // threads per threadgroup = SG * 32

    // Map linear index → (col_tile, row_tile) via Morton (Z-curve) ordering.
    uint col_tile = morton_x(tgid_linear);
    uint row_tile = morton_y(tgid_linear);

    auto tensorA = tensor(A, dextents<int,2>{(int)K, (int)M}, array<int,2>{1, (int)K});
    auto tensorB = tensor(B, dextents<int,2>{(int)N, (int)K}, array<int,2>{1, (int)N});
    auto tensorC = tensor(C, dextents<int,2>{(int)N, (int)M}, array<int,2>{1, (int)N});

    auto mA = tensorA.slice(0,                    (int)row_tile * M_TILE);
    auto mB = tensorB.slice((int)col_tile * N_TILE, 0);
    auto mC = tensorC.slice((int)col_tile * N_TILE, (int)row_tile * M_TILE);

    constexpr auto desc = matmul2d_descriptor(M_TILE, N_TILE, static_cast<int>(metal::dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SG>> op;

    op.run(mA, mB, mC);
}
