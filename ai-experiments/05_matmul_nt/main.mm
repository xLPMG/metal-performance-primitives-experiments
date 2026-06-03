// 05_matmul_nt – host driver
// NT: A[M×K] @ B^T where B is stored as K×N (each row = one output neuron's weights).
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
#define METALLIB_PATH "build/05_matmul_nt/kernel.metallib"
#endif

static id<MTLDevice>               gDevice;
static id<MTLCommandQueue>         gQueue;
static id<MTLComputePipelineState> gPSO;

static void setup() {
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue  = [gDevice newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB_PATH] error:&err];
    if (!lib) { NSLog(@"Library: %@", err); exit(1); }
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_nt"];
    if (!fn)  { puts("Function not found"); exit(1); }
    gPSO = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gPSO) { NSLog(@"PSO: %@", err); exit(1); }
}

// CPU ref for NT: C[m,n] = sum_k A[m,k] * B[n,k]
// B stored N×K row-major: B[n,k] at n*K+k  (mirrors Metal extents {K,N} strides {1,K})
static void cpu_ref(const std::vector<uint16_t>& A,
                    const std::vector<uint16_t>& Bstored,
                    std::vector<float>& C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += f16_to_f32(A[m*K+k]) * f16_to_f32(Bstored[n*K+k]);
            C[m*N+n] = acc;
        }
}

struct Bufs { id<MTLBuffer> A, B, C, p; };

static Bufs makeBuffers(int M, int N, int K,
                        const std::vector<uint16_t>& hA,
                        const std::vector<uint16_t>& hB) {
    auto sh = [&](const void* d, size_t n){
        return [gDevice newBufferWithBytes:d length:n options:MTLResourceStorageModeShared]; };
    Bufs b;
    b.A = sh(hA.data(), M*K*2);
    b.B = sh(hB.data(), K*N*2);
    b.C = [gDevice newBufferWithLength:M*N*4 options:MTLResourceStorageModeShared];
    uint32_t pv[3] = {(uint32_t)M,(uint32_t)N,(uint32_t)K};
    b.p = sh(pv, 12);
    return b;
}

static id<MTLCommandBuffer> dispatch(const Bufs& b, int M, int N) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:gPSO];
    [enc setBuffer:b.A offset:0                  atIndex:0];
    [enc setBuffer:b.B offset:0                  atIndex:1];
    [enc setBuffer:b.C offset:0                  atIndex:2];
    [enc setBuffer:b.p offset:0*sizeof(uint32_t) atIndex:3];
    [enc setBuffer:b.p offset:1*sizeof(uint32_t) atIndex:4];
    [enc setBuffer:b.p offset:2*sizeof(uint32_t) atIndex:5];
    NSUInteger w = gPSO.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake((N+31)/32,(M+63)/64,1)
        threadsPerThreadgroup:MTLSizeMake(w*4,1,1)];
    [enc endEncoding]; [cb commit]; return cb;
}

int main() {
    setup();
    printf("GPU: %s\n", [gDevice.name UTF8String]);
    {
        const int M=128,N=64,K=128;
        std::vector<uint16_t> hA(M*K), hB(K*N);
        for (int i=0;i<M*K;++i) hA[i]=f32_to_f16((i%7)*0.1f);
        for (int i=0;i<K*N;++i) hB[i]=f32_to_f16((i%5)*0.2f);
        std::vector<float> ref(M*N);
        cpu_ref(hA,hB,ref,M,N,K);
        Bufs b = makeBuffers(M,N,K,hA,hB);
        [dispatch(b,M,N) waitUntilCompleted];
        const float* gpu = (const float*)[b.C contents];
        float maxErr=0.f;
        for (int i=0;i<M*N;++i) maxErr=std::max(maxErr,std::abs(gpu[i]-ref[i]));
        printf("Correctness (M=%d N=%d K=%d): max_err=%.4f  %s\n",M,N,K,maxErr,maxErr<0.5f?"PASS":"FAIL");
    }
    {
        const int M=2048,N=2048,K=2048;
        std::vector<uint16_t> hA(M*K),hB(K*N);
        for (int i=0;i<M*K;++i) hA[i]=f32_to_f16((i%7)*0.1f);
        for (int i=0;i<K*N;++i) hB[i]=f32_to_f16((i%5)*0.2f);
        Bufs b = makeBuffers(M,N,K,hA,hB);
        auto r = bench(10,100,[&]{return dispatch(b,M,N);});
        r.print("05 matmul half×half→float (NT)", 2.0*M*N*K);
    }
    return 0;
}
