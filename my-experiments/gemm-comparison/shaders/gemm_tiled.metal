#include <metal_stdlib>
using namespace metal;

kernel void gemm_tiled(   device const float* A [[buffer(0)]],
                            device const float* B [[buffer(1)]],
                            device float* C [[buffer(2)]],
                            constant uint& N [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 tid [[thread_position_in_threadgroup]])
{
    constexpr uint TILE = 16;
    uint numTiles = (N + TILE - 1) / TILE;

    // Shared memory for tiles of A and B among threads in the same threadgroup
    threadgroup float Asub[TILE][TILE];
    threadgroup float Bsub[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;

    float acc = 0.0f;

    for (uint t = 0; t < numTiles; t++)
    {
        // Load tile of A
        // zero padding for out-of-bounds
        uint aCol = t * TILE + tid.x;
        Asub[tid.y][tid.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;

        // Load tile of B
        uint bRow = t * TILE + tid.y;
        Bsub[tid.y][tid.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        // Wait for all threads to finish loading their tiles into shared memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial product for this tile
        for (uint k = 0; k < TILE; k++)
        {
            acc += Asub[tid.y][k] * Bsub[k][tid.x];
        }

        // Wait for all threads to finish computing with the current tile before loading the next one
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store C
    // guard handles the case when N is not a multiple of tile size
    // because some threads are launched even for out-of-bounds elements
    if (row < N && col < N)
    {
        C[row * N + col] = acc;
    }
}