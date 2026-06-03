// 01_matmul_hh_h – host driver
// Verifies half×half→half matmul against a CPU reference, then benchmarks.
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
#define METALLIB_PATH "build/01_matmul_hh_h/kernel.metallib"
#endif

static id<MTLDevice>               gDevice;
static id<MTLCommandQueue>         gQueue;
static id<MTLComputePipelineState> gPSO;

static void setup() {
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue  = [gDevice newCommandQueue];
    NSError* err = nil;
    NSString* path = @METALLIB_PATH;
    id<MTLLibrary> lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) { NSLog(@"Library load failed: %@", err); exit(1); }
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_hh_h"];
    if (!fn)  { puts("Function not found"); exit(1); }
    gPSO = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gPSO) { NSLog(@"PSO failed: %@", err); exit(1); }
}

// CPU reference  C[m,n] = Σ_k A[m,k]*B[k,n]  (in float32, store as half)
static void cpu_ref(const std::vector<uint16_t>& A, const std::vector<uint16_t>& B,
                    std::vector<uint16_t>& C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += f16_to_f32(A[m*K+k]) * f16_to_f32(B[k*N+n]);
            C[m*N+n] = f32_to_f16(acc);
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
    b.C = [gDevice newBufferWithLength:M*N*2 options:MTLResourceStorageModeShared];
    uint32_t p[3] = {(uint32_t)M,(uint32_t)N,(uint32_t)K};
    b.params = mkShared(p, 12);
    return b;
}

static id<MTLCommandBuffer> dispatch(const Bufs& b, int M, int N) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gPSO];
    [enc setBuffer:b.A      offset:0                 atIndex:0];
    [enc setBuffer:b.B      offset:0                 atIndex:1];
    [enc setBuffer:b.C      offset:0                 atIndex:2];
    [enc setBuffer:b.params offset:0*sizeof(uint32_t) atIndex:3];
    [enc setBuffer:b.params offset:1*sizeof(uint32_t) atIndex:4];
    [enc setBuffer:b.params offset:2*sizeof(uint32_t) atIndex:5];
    NSUInteger simdW = gPSO.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake((N+31)/32, (M+63)/64, 1)
        threadsPerThreadgroup:MTLSizeMake(simdW*4, 1, 1)];
    [enc endEncoding];
    [cb commit];
    return cb;
}

int main() {
    setup();
    printf("GPU: %s\n", [gDevice.name UTF8String]);

    // ── correctness (small) ──────────────────────────────────────────────
    {
        const int M=128, N=64, K=128;
        std::vector<uint16_t> hA(M*K), hB(K*N), hC_ref(M*N);
        for (int i = 0; i < M*K; ++i) hA[i] = f32_to_f16((i%7)*0.1f);
        for (int i = 0; i < K*N; ++i) hB[i] = f32_to_f16((i%5)*0.2f);
        cpu_ref(hA, hB, hC_ref, M, N, K);

        Bufs b = makeBuffers(M, N, K, hA, hB);
        id<MTLCommandBuffer> cb = dispatch(b, M, N);
        [cb waitUntilCompleted];

        const uint16_t* gpu = (const uint16_t*)[b.C contents];
        float maxErr = 0.f;
        for (int i = 0; i < M*N; ++i)
            maxErr = std::max(maxErr, std::abs(f16_to_f32(gpu[i]) - f16_to_f32(hC_ref[i])));
        printf("Correctness (M=%d N=%d K=%d): max_err=%.4f  %s\n",
               M, N, K, maxErr, maxErr < 0.5f ? "PASS" : "FAIL");
    }

    // ── benchmark (large) ───────────────────────────────────────────────
    {
        const int M=2048, N=2048, K=2048;
        std::vector<uint16_t> hA(M*K), hB(K*N);
        for (int i = 0; i < M*K; ++i) hA[i] = f32_to_f16((i%7)*0.1f);
        for (int i = 0; i < K*N; ++i) hB[i] = f32_to_f16((i%5)*0.2f);
        Bufs b = makeBuffers(M, N, K, hA, hB);

        auto result = bench(10, 100, [&]{ return dispatch(b, M, N); });
        result.print("01 matmul half×half→half", 2.0*M*N*K);
    }
    return 0;
}
