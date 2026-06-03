// 04_matmul_i8i8_i32 – host driver
// int8 × int8 → int32. Integer GEMM; no floating-point conversion needed.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "common/bench.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdint>

#ifndef METALLIB_PATH
#define METALLIB_PATH "build/04_matmul_i8i8_i32/kernel.metallib"
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
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_i8i8_i32"];
    if (!fn)  { puts("Function not found"); exit(1); }
    gPSO = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gPSO) { NSLog(@"PSO: %@", err); exit(1); }
}

static void cpu_ref(const std::vector<int8_t>& A, const std::vector<int8_t>& B,
                    std::vector<int32_t>& C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)A[m*K+k] * (int32_t)B[k*N+n];
            C[m*N+n] = acc;
        }
}

struct Bufs { id<MTLBuffer> A, B, C, p; };

static Bufs makeBuffers(int M, int N, int K,
                        const std::vector<int8_t>& hA,
                        const std::vector<int8_t>& hB) {
    auto sh = [&](const void* d, size_t n){
        return [gDevice newBufferWithBytes:d length:n options:MTLResourceStorageModeShared]; };
    Bufs b;
    b.A = sh(hA.data(), M*K);
    b.B = sh(hB.data(), K*N);
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
        std::vector<int8_t> hA(M*K), hB(K*N);
        for (int i=0;i<M*K;++i) hA[i]=(int8_t)((i%15)-7);
        for (int i=0;i<K*N;++i) hB[i]=(int8_t)((i%11)-5);
        std::vector<int32_t> ref(M*N);
        cpu_ref(hA,hB,ref,M,N,K);
        Bufs b = makeBuffers(M,N,K,hA,hB);
        [dispatch(b,M,N) waitUntilCompleted];
        const int32_t* gpu = (const int32_t*)[b.C contents];
        int32_t maxErr=0;
        for (int i=0;i<M*N;++i) maxErr=std::max(maxErr,std::abs(gpu[i]-ref[i]));
        printf("Correctness (M=%d N=%d K=%d): max_err=%d  %s\n",M,N,K,maxErr,maxErr==0?"PASS":"FAIL");
    }
    {
        const int M=2048,N=2048,K=2048;
        std::vector<int8_t> hA(M*K),hB(K*N);
        for (int i=0;i<M*K;++i) hA[i]=(int8_t)((i%15)-7);
        for (int i=0;i<K*N;++i) hB[i]=(int8_t)((i%11)-5);
        Bufs b = makeBuffers(M,N,K,hA,hB);
        auto r = bench(10,100,[&]{return dispatch(b,M,N);});
        r.print("04 matmul int8×int8→int32", 2.0*M*N*K);
    }
    return 0;
}
