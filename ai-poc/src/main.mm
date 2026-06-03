#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// float16 helpers via clang's __fp16 (supported by Apple clang on arm64)
// ---------------------------------------------------------------------------
static uint16_t f32_to_f16(float v) {
    __fp16 h = static_cast<__fp16>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof bits);
    return bits;
}

static float f16_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof h);
    return static_cast<float>(h);
}

// ---------------------------------------------------------------------------
// CPU reference: C[m,n] = sum_k A[m,k] * B[k,n]
// ---------------------------------------------------------------------------
static void cpu_matmul(const std::vector<uint16_t>& A,
                       const std::vector<uint16_t>& B,
                       std::vector<float>&           C,
                       int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += f16_to_f32(A[m * K + k]) * f16_to_f32(B[k * N + n]);
            }
            C[m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Dimensions – multiples of the tile (m=64, n=32) for a clean dispatch.
    const int M = 128;   // rows of A and C
    const int N =  64;   // cols of B and C
    const int K = 128;   // cols of A / rows of B

    // -----------------------------------------------------------------------
    // Set up Metal device and pipeline
    // -----------------------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "No Metal device found\n";
        return 1;
    }
    std::cout << "GPU: " << [device.name UTF8String] << "\n";

    NSError* error = nil;

    // The metallib is compiled by the Makefile and placed next to the binary.
    NSString* metallibPath = @"build/kernels.metallib";
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:metallibPath]
                                                 error:&error];
    if (!library) {
        NSLog(@"Failed to load metallib at '%@': %@", metallibPath, error);
        return 1;
    }

    id<MTLFunction> fn = [library newFunctionWithName:@"matmul_mpp"];
    if (!fn) {
        std::cerr << "Kernel function 'matmul_mpp' not found in library\n";
        return 1;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        NSLog(@"Failed to create pipeline: %@", error);
        return 1;
    }

    // -----------------------------------------------------------------------
    // Prepare host data
    // -----------------------------------------------------------------------
    std::vector<uint16_t> hA(M * K), hB(K * N);
    std::vector<uint16_t> hC_gpu(M * N, 0);   // half output from GPU
    std::vector<float>    hC_cpu(M * N, 0.f);

    // Simple deterministic fill.
    for (int i = 0; i < M * K; ++i) hA[i] = f32_to_f16(static_cast<float>(i % 7) * 0.1f);
    for (int i = 0; i < K * N; ++i) hB[i] = f32_to_f16(static_cast<float>(i % 5) * 0.2f);

    // -----------------------------------------------------------------------
    // Create Metal buffers (shared memory – visible to both CPU and GPU)
    // -----------------------------------------------------------------------
    auto mkBuf = [&](const void* bytes, size_t len) {
        return [device newBufferWithBytes:bytes length:len
                                  options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> bufA = mkBuf(hA.data(), M * K * sizeof(uint16_t));
    id<MTLBuffer> bufB = mkBuf(hB.data(), K * N * sizeof(uint16_t));
    id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(uint16_t)
                                             options:MTLResourceStorageModeShared];

    // Zero C on the GPU side.
    std::memset([bufC contents], 0, M * N * sizeof(uint16_t));

    uint32_t uM = M, uN = N, uK = K;
    id<MTLBuffer> bufParams = [device newBufferWithLength:3 * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    uint32_t* params = static_cast<uint32_t*>([bufParams contents]);
    params[0] = uM; params[1] = uN; params[2] = uK;

    // -----------------------------------------------------------------------
    // Encode and submit
    // -----------------------------------------------------------------------
    id<MTLCommandQueue>   queue  = [device newCommandQueue];
    id<MTLCommandBuffer>  cmdbuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA      offset:0               atIndex:0];
    [enc setBuffer:bufB      offset:0               atIndex:1];
    [enc setBuffer:bufC      offset:0               atIndex:2];
    [enc setBuffer:bufParams offset:0 * sizeof(uint32_t) atIndex:3];
    [enc setBuffer:bufParams offset:1 * sizeof(uint32_t) atIndex:4];
    [enc setBuffer:bufParams offset:2 * sizeof(uint32_t) atIndex:5];

    // Each threadgroup has 4 simdgroups.
    NSUInteger simdW        = pipeline.threadExecutionWidth;         // 32
    MTLSize threadsPerTG    = MTLSizeMake(simdW * 4, 1, 1);         // 128 threads
    // tgid.x covers N in steps of 32; tgid.y covers M in steps of 64.
    MTLSize threadgroupCount = MTLSizeMake((N + 31) / 32,
                                           (M + 63) / 64,
                                           1);

    [enc dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadsPerTG];
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // -----------------------------------------------------------------------
    // Read back and compare against CPU reference
    // -----------------------------------------------------------------------
    // GPU result is in half; convert to float for comparison.
    const uint16_t* gpu_raw = static_cast<const uint16_t*>([bufC contents]);
    for (int i = 0; i < M * N; ++i)
        hC_gpu[i] = gpu_raw[i];

    cpu_matmul(hA, hB, hC_cpu, M, N, K);

    float maxErr = 0.f;
    for (int i = 0; i < M * N; ++i)
        maxErr = std::max(maxErr, std::abs(f16_to_f32(hC_gpu[i]) - hC_cpu[i]));

    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "C[0,0]  GPU=" << f16_to_f32(hC_gpu[0])   << "  CPU=" << hC_cpu[0]   << "\n";
    std::cout << "C[1,0]  GPU=" << f16_to_f32(hC_gpu[N])   << "  CPU=" << hC_cpu[N]   << "\n";
    std::cout << "C[0,1]  GPU=" << f16_to_f32(hC_gpu[1])   << "  CPU=" << hC_cpu[1]   << "\n";
    std::cout << "Max absolute error: " << maxErr << "\n";
    // Use a larger tolerance since half has ~3 decimal digits of precision.
    std::cout << (maxErr < 0.5f ? "PASS\n" : "FAIL\n");

    return maxErr < 0.5f ? 0 : 1;
}
