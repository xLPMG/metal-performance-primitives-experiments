// 08_conv2d_basic – host driver
// Verifies the 3×3 NHWC conv2d output against a CPU reference.
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
#define METALLIB_PATH "build/08_conv2d_basic/kernel.metallib"
#endif

static const int BATCH=1, IN_H=8, IN_W=8, OUT_H=6, OUT_W=6;
static const int C_IN=16, C_OUT=32, KH=3, KW=3;

static id<MTLDevice>               gDevice;
static id<MTLCommandQueue>         gQueue;
static id<MTLComputePipelineState> gPSO;

static void setup() {
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue  = [gDevice newCommandQueue];
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB_PATH] error:&err];
    if (!lib) { NSLog(@"Library: %@", err); exit(1); }
    id<MTLFunction> fn = [lib newFunctionWithName:@"conv2d_basic"];
    if (!fn)  { puts("Function not found"); exit(1); }
    gPSO = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    if (!gPSO) { NSLog(@"PSO: %@", err); exit(1); }
}

// CPU reference: NHWC activation × HWIO weights → NHWO output
static void cpu_ref(const std::vector<uint16_t>& act,
                    const std::vector<uint16_t>& wgt,
                    std::vector<float>& dst) {
    for (int oh = 0; oh < OUT_H; ++oh)
    for (int ow = 0; ow < OUT_W; ++ow)
    for (int oc = 0; oc < C_OUT; ++oc) {
        float acc = 0.f;
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw)
        for (int ic = 0; ic < C_IN; ++ic) {
            int ih = oh + kh, iw = ow + kw;
            // act: NHWC [n=0, ih, iw, ic]
            float a = f16_to_f32(act[ih*IN_W*C_IN + iw*C_IN + ic]);
            // wgt: HWIO [kh, kw, ic, oc]
            float w = f16_to_f32(wgt[kh*KW*C_IN*C_OUT + kw*C_IN*C_OUT + ic*C_OUT + oc]);
            acc += a * w;
        }
        // dst: NHWO [n=0, oh, ow, oc]
        dst[oh*OUT_W*C_OUT + ow*C_OUT + oc] = acc;
    }
}

int main() {
    setup();
    printf("GPU: %s\n", [gDevice.name UTF8String]);

    const size_t actElems = BATCH*IN_H*IN_W*C_IN;
    const size_t wgtElems = KH*KW*C_IN*C_OUT;
    const size_t dstElems = BATCH*OUT_H*OUT_W*C_OUT;

    std::vector<uint16_t> hAct(actElems), hWgt(wgtElems);
    for (size_t i=0;i<actElems;++i) hAct[i]=f32_to_f16((i%7)*0.1f);
    for (size_t i=0;i<wgtElems;++i) hWgt[i]=f32_to_f16((i%5)*0.2f - 0.4f);

    std::vector<float> ref(dstElems);
    cpu_ref(hAct, hWgt, ref);

    auto sh = [&](const void* d, size_t n){
        return [gDevice newBufferWithBytes:d length:n options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> bufAct = sh(hAct.data(), actElems*2);
    id<MTLBuffer> bufWgt = sh(hWgt.data(), wgtElems*2);
    id<MTLBuffer> bufDst = [gDevice newBufferWithLength:dstElems*2
                                                options:MTLResourceStorageModeShared];

    auto dispatchConv = [&]() -> id<MTLCommandBuffer> {
        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gPSO];
        [enc setBuffer:bufAct offset:0 atIndex:0];
        [enc setBuffer:bufWgt offset:0 atIndex:1];
        [enc setBuffer:bufDst offset:0 atIndex:2];
        // One threadgroup per output spatial position (OUT_W × OUT_H).
        NSUInteger w = gPSO.threadExecutionWidth;
        [enc dispatchThreadgroups:MTLSizeMake(OUT_W, OUT_H, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        [enc endEncoding]; [cb commit]; return cb;
    };

    [dispatchConv() waitUntilCompleted];

    const uint16_t* gpu = (const uint16_t*)[bufDst contents];
    float maxErr = 0.f;
    for (size_t i=0;i<dstElems;++i)
        maxErr = std::max(maxErr, std::abs(f16_to_f32(gpu[i]) - ref[i]));
    printf("Correctness (1×%d×%d×%d, 3×3, →%d): max_err=%.4f  %s\n",
           IN_H, IN_W, C_IN, C_OUT, maxErr, maxErr < 0.5f ? "PASS" : "FAIL");

    auto r = bench(10, 100, dispatchConv);
    // FLOPs: 2 * OUT_H * OUT_W * KH * KW * C_IN * C_OUT * BATCH
    double flops = 2.0 * OUT_H * OUT_W * KH * KW * C_IN * C_OUT * BATCH;
    r.print("08 conv2d 3×3 NHWC half→half", flops);

    return 0;
}
