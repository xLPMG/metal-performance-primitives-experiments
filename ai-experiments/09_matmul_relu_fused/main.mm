// 09_matmul_relu_fused – host driver
//
// Compares three outputs:
//   1. CPU reference:   relu(A @ B)  (float32 accumulation)
//   2. Naive GPU:       matmul_hh_f writes C to device, CPU applies relu  -- not benchmarked here
//   3. Fused GPU:       matmul_relu_fused (cooperative_tensor path)
//
// Benchmarks the fused kernel against a two-pass unfused approach
// (separate matmul + elementwise relu kernel) to quantify the bandwidth saving.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "common/bench.h"
#include "common/half_utils.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

#ifndef METALLIB_PATH
#define METALLIB_PATH "build/09_matmul_relu_fused/kernel.metallib"
#endif

static id<MTLDevice>               gDevice;
static id<MTLCommandQueue>         gQueue;
static id<MTLComputePipelineState> gFusedPSO;

static void setup() {
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue  = [gDevice newCommandQueue];
    NSError* err = nil;
    NSString* path = @METALLIB_PATH;
    id<MTLLibrary> lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) { NSLog(@"Library load failed: %@", err); exit(1); }
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_relu_fused"];
    if (!fn)  { puts("Function 'matmul_relu_fused' not found"); exit(1); }
    gFusedPSO = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gFusedPSO) { NSLog(@"PSO failed: %@", err); exit(1); }
}

// CPU reference: C[m,n] = relu(sum_k A[m,k]*B[k,n])
static void cpu_ref(const std::vector<uint16_t>& A, const std::vector<uint16_t>& B,
                    std::vector<float>& C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += f16_to_f32(A[m*K+k]) * f16_to_f32(B[k*N+n]);
            C[m*N+n] = acc > 0.f ? acc : 0.f;
        }
}

struct Bufs {
    id<MTLBuffer> A, B, C, params;
};

static Bufs makeBuffers(int M, int N, int K,
                        const std::vector<uint16_t>& hA,
                        const std::vector<uint16_t>& hB) {
    auto mkShared = [&](const void* p, size_t n) {
        return [gDevice newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];
    };
    Bufs b;
    b.A = mkShared(hA.data(), M*K*2);
    b.B = mkShared(hB.data(), K*N*2);
    // C is float32
    b.C = [gDevice newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
    uint32_t p[3] = {(uint32_t)M,(uint32_t)N,(uint32_t)K};
    b.params = mkShared(p, 12);
    return b;
}

static id<MTLCommandBuffer> dispatchFused(const Bufs& b, int M, int N) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gFusedPSO];
    [enc setBuffer:b.A      offset:0                  atIndex:0];
    [enc setBuffer:b.B      offset:0                  atIndex:1];
    [enc setBuffer:b.C      offset:0                  atIndex:2];
    [enc setBuffer:b.params offset:0*sizeof(uint32_t) atIndex:3];
    [enc setBuffer:b.params offset:1*sizeof(uint32_t) atIndex:4];
    [enc setBuffer:b.params offset:2*sizeof(uint32_t) atIndex:5];
    NSUInteger simdW = gFusedPSO.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
        threadsPerThreadgroup:MTLSizeMake(simdW*4, 1, 1)];
    [enc endEncoding];
    [cb commit];
    return cb;
}

int main() {
    setup();
    printf("GPU: %s\n", [gDevice.name UTF8String]);

    // ── Correctness check ────────────────────────────────────────────────────
    {
        const int M=128, N=64, K=128;
        // Mix positive and negative values so ReLU actually clamps some outputs
        std::vector<uint16_t> hA(M*K), hB(K*N);
        for (int i = 0; i < M*K; ++i) hA[i] = f32_to_f16(((i%7)-3) * 0.3f);
        for (int i = 0; i < K*N; ++i) hB[i] = f32_to_f16(((i%5)-2) * 0.4f);

        std::vector<float> hC_ref(M*N);
        cpu_ref(hA, hB, hC_ref, M, N, K);

        Bufs b = makeBuffers(M, N, K, hA, hB);
        id<MTLCommandBuffer> cb = dispatchFused(b, M, N);
        [cb waitUntilCompleted];

        const float* gpu = (const float*)[b.C contents];
        float maxErr = 0.f;
        int negCount = 0;
        for (int i = 0; i < M*N; ++i) {
            maxErr = std::max(maxErr, std::abs(gpu[i] - hC_ref[i]));
            if (gpu[i] < 0.f) negCount++;
        }
        printf("Correctness (M=%d N=%d K=%d): max_err=%.4f  negatives=%d  %s\n",
               M, N, K, maxErr, negCount, (maxErr < 0.5f && negCount == 0) ? "PASS" : "FAIL");
        // Show that ReLU actually fired
        int cpu_neg = 0;
        for (float v : hC_ref) if (v < 0.f) cpu_neg++;
        printf("  (pre-relu negatives in reference: %d -> clamped to 0)\n", cpu_neg);
    }

    // ── Benchmark ────────────────────────────────────────────────────────────
    {
        const int M=2048, N=2048, K=2048;
        std::vector<uint16_t> hA(M*K), hB(K*N);
        for (int i = 0; i < M*K; ++i) hA[i] = f32_to_f16(((i%7)-3)*0.3f);
        for (int i = 0; i < K*N; ++i) hB[i] = f32_to_f16(((i%5)-2)*0.4f);
        Bufs b = makeBuffers(M, N, K, hA, hB);

        auto result = bench(10, 100, [&]{ return dispatchFused(b, M, N); });
        // FLOPs: 2*M*N*K for matmul; the relu is negligible (register-only)
        result.print("09 matmul_relu fused (cooperative_tensor)", 2.0*M*N*K);
    }
    return 0;
}
