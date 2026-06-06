#!/usr/bin/env python3
"""Generate one .metal shader per (M_tile, N_tile, simdgroups) combination."""

import os, itertools

TILES    = [(32,16), (32,32), (64,32), (64,64), (128,32), (128,64), (128,128)]
SIMDGRPS = list(range(1, 33))  # 1 to 32

TEMPLATE = """\
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using metal::array;
using metal::dextents;
using metal::tensor;
using mpp::tensor_ops::matmul2d_descriptor;
using mpp::tensor_ops::matmul2d;

kernel void gemm_mpp(
    device half* A [[buffer(0)]],
    device half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{{
    constexpr int M_TILE = {m_tile};
    constexpr int N_TILE = {n_tile};

    auto tensorA = tensor(A, dextents<int,2>{{(int)K, (int)M}}, array<int,2>{{1, (int)K}});
    auto tensorB = tensor(B, dextents<int,2>{{(int)N, (int)K}}, array<int,2>{{1, (int)N}});
    auto tensorC = tensor(C, dextents<int,2>{{(int)N, (int)M}}, array<int,2>{{1, (int)N}});

    auto mA = tensorA.slice(0,                    (int)tgid.y * M_TILE);
    auto mB = tensorB.slice((int)tgid.x * N_TILE, 0);
    auto mC = tensorC.slice((int)tgid.x * N_TILE, (int)tgid.y * M_TILE);

    constexpr auto desc = matmul2d_descriptor(M_TILE, N_TILE, static_cast<int>(metal::dynamic_extent));
    matmul2d<desc, metal::execution_simdgroups<{sg}>> op;
    op.run(mA, mB, mC);
}}
"""

os.makedirs("shaders", exist_ok=True)

for (m, n), sg in itertools.product(TILES, SIMDGRPS):
    name = f"{m}x{n}_sg{sg}"
    path = f"shaders/mpp_{name}.metal"
    with open(path, "w") as f:
        f.write(TEMPLATE.format(m_tile=m, n_tile=n, sg=sg))

print(f"Generated {len(TILES) * len(SIMDGRPS)} shaders.")  # 7 tiles x 32 = 224
