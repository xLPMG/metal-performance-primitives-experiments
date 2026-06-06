#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <cstddef>

#ifndef METALLIB_PATH
#define METALLIB_PATH "build/kernels.metallib"
#endif

// handle to the GPU
static id<MTLDevice> device;
// FIFO command queue for submitting work to the GPU
static id<MTLCommandQueue> commandQueue;

// pipeline states
// = compiled GPU programs that can be executed
static id<MTLComputePipelineState> naivePipeline = nil;
static id<MTLComputePipelineState> tiledPipeline = nil;

static id<MTLComputePipelineState> load_pipeline(const char *name, id<MTLLibrary> lib)
{
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { NSLog(@"Function %s not found", name); exit(1); }
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { NSLog(@"PSO: %@", err); exit(1); }
    return pso;
}

void metal_init()
{
    device = MTLCreateSystemDefaultDevice();
    if (!device)
    {
        NSLog(@"Metal is not supported on this device");
        exit(1);
    }
    commandQueue = [device newCommandQueue];

    NSError *err = nil;
    // load the default library (compiled from .metal files in the project)
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB_PATH] error:&err];
    if (!lib) { NSLog(@"Library: %@", err); exit(1); }

    // create pipelines for our kernels
    naivePipeline = load_pipeline("gemm_naive", lib);
    tiledPipeline = load_pipeline("gemm_tiled", lib);
}

void metal_shutdown()
{
    device = nil;
    commandQueue = nil;
    naivePipeline = nil;
    tiledPipeline = nil;
}

void metal_gemm_naive_run(float *A, float *B, float *C, int N)
{
    size_t bytes = N * N * sizeof(float);

    id<MTLBuffer> bufA = [device newBufferWithBytes:A length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithBytes:B length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufN = [device newBufferWithBytes:&N length:sizeof(int) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:naivePipeline];

    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufN offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake(N, N, 1);
    MTLSize tpg = MTLSizeMake(16, 16, 1);
 
    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(C, [bufC contents], bytes);
}

void metal_gemm_naive(float *A, float *B, float *C, int N)
{
    static bool initialized = false;
    if (!initialized) {
        metal_init();
        initialized = true;
    }
    metal_gemm_naive_run(A, B, C, N);
}

void metal_gemm_tiled_run(float *A, float *B, float *C, int N)
{
    size_t bytes = N * N * sizeof(float);

    id<MTLBuffer> bufA = [device newBufferWithBytes:A length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithBytes:B length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufN = [device newBufferWithBytes:&N length:sizeof(int) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:tiledPipeline];

    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufN offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake(N, N, 1);
    MTLSize tpg  = MTLSizeMake(16, 16, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(C, [bufC contents], bytes);
}

void metal_gemm_tiled(float *A, float *B, float *C, int N)
{
    static bool initialized = false;
    if (!initialized) {
        metal_init();
        initialized = true;
    }
    metal_gemm_tiled_run(A, B, C, N);
}