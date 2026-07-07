#import "metal_bridge.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
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
static id<MTLComputePipelineState> mppPipeline   = nil;
static id<MTLComputePipelineState> mortonPipeline = nil;

static id<MTLComputePipelineState> load_pipeline(const char *name, id<MTLLibrary> lib)
{
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn)
    {
        NSLog(@"Function %s not found", name);
        exit(1);
    }
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
    {
        NSLog(@"PSO: %@", err);
        exit(1);
    }
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
    if (!lib)
    {
        NSLog(@"Library: %@", err);
        exit(1);
    }

    // create pipelines for our kernels
    naivePipeline  = load_pipeline("gemm_naive",      lib);
    tiledPipeline  = load_pipeline("gemm_tiled",      lib);
    mppPipeline    = load_pipeline("gemm_mpp",        lib);
    mortonPipeline = load_pipeline("gemm_mpp_morton", lib);
}

void metal_shutdown()
{
    device = nil;
    commandQueue = nil;
    naivePipeline = nil;
    tiledPipeline = nil;
    mppPipeline   = nil;
    mortonPipeline = nil;
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
    if (!initialized)
    {
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
    MTLSize tpg = MTLSizeMake(16, 16, 1);

    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(C, [bufC contents], bytes);
}

void metal_gemm_tiled(float *A, float *B, float *C, int N)
{
    static bool initialized = false;
    if (!initialized)
    {
        metal_init();
        initialized = true;
    }
    metal_gemm_tiled_run(A, B, C, N);
}

void metal_gemm_mps(float *A, float *B, float *C, int N)
{
    static bool initialized = false;
    // describes the shape and type of the matrices
    // rows, columns, stride (in bytes), data type
    static MPSMatrixDescriptor *desc = nil;
    // the precompiled, tuned GEMM kernel
    static MPSMatrixMultiplication *matmul = nil;

    if (!initialized)
    {
        metal_init();
        desc = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:N rowBytes:N * sizeof(float)
                                                    dataType:MPSDataTypeFloat32];
        matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device resultRows:N resultColumns:N interiorColumns:N];
        initialized = true;
    }

    size_t bytes = N * N * sizeof(float);

    id<MTLBuffer> bufA = [device newBufferWithBytes:A length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithBytes:B length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    MPSMatrix *mpsA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];
    MPSMatrix *mpsB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:desc];
    MPSMatrix *mpsC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:desc];

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    [matmul encodeToCommandBuffer:cmd leftMatrix:mpsA rightMatrix:mpsB resultMatrix:mpsC];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(C, [bufC contents], bytes);
}

// f32 -> f16 conversion (software, host-side)
static uint16_t f32_to_f16(float f)
{
    // Use __fp16 if available on this toolchain
    __fp16 h = (__fp16)f;
    uint16_t bits;
    __builtin_memcpy(&bits, &h, 2);
    return bits;
}

static float f16_to_f32(uint16_t bits)
{
    __fp16 h;
    __builtin_memcpy(&h, &bits, 2);
    return (float)h;
}

void metal_gemm_mps_f16(float *A, float *B, float *C, int N)
{
    static bool initialized = false;
    static MPSMatrixDescriptor *desc = nil;
    static MPSMatrixMultiplication *matmul = nil;

    if (!initialized)
    {
        metal_init();
        // Same setup as the f32 variant but with MPSDataTypeFloat16 and 2-byte row stride
        desc = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:N rowBytes:N * sizeof(uint16_t)
                                                    dataType:MPSDataTypeFloat16];
        matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device resultRows:N resultColumns:N interiorColumns:N];
        initialized = true;
    }

    size_t elems = (size_t)N * N;
    size_t bytes = elems * sizeof(uint16_t);

    // Convert f32 inputs to f16
    id<MTLBuffer> bufA = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];

    uint16_t *pA = (uint16_t *)[bufA contents];
    uint16_t *pB = (uint16_t *)[bufB contents];
    for (size_t i = 0; i < elems; i++) pA[i] = f32_to_f16(A[i]);
    for (size_t i = 0; i < elems; i++) pB[i] = f32_to_f16(B[i]);

    MPSMatrix *mpsA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:desc];
    MPSMatrix *mpsB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:desc];
    MPSMatrix *mpsC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:desc];

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    [matmul encodeToCommandBuffer:cmd leftMatrix:mpsA rightMatrix:mpsB resultMatrix:mpsC];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Convert f16 output back to f32
    uint16_t *pC = (uint16_t *)[bufC contents];
    for (size_t i = 0; i < elems; i++) C[i] = f16_to_f32(pC[i]);
}

void metal_gemm_mpp(float *A, float *B, float *C, int M, int N, int K)
{
    static bool initialized = false;
    if (!initialized) { metal_init(); initialized = true; }

    size_t elems   = (size_t)M * K;
    size_t bytesA  = elems * sizeof(uint16_t);
    size_t bytesB  = (size_t)K * N * sizeof(uint16_t);
    size_t bytesC  = (size_t)M * N * sizeof(uint16_t);

    // Convert f32 inputs to f16
    id<MTLBuffer> bufA = [device newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];

    uint16_t *pA = (uint16_t *)[bufA contents];
    uint16_t *pB = (uint16_t *)[bufB contents];
    for (size_t i = 0; i < (size_t)M * K; i++) pA[i] = f32_to_f16(A[i]);
    for (size_t i = 0; i < (size_t)K * N; i++) pB[i] = f32_to_f16(B[i]);

    // M, N, K as uint buffers
    uint32_t uM = M, uN = N, uK = K;
    id<MTLBuffer> bufM = [device newBufferWithBytes:&uM length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufN = [device newBufferWithBytes:&uN length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [device newBufferWithBytes:&uK length:4 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:mppPipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufM offset:0 atIndex:3];
    [enc setBuffer:bufN offset:0 atIndex:4];
    [enc setBuffer:bufK offset:0 atIndex:5];

    // Must match M_TILE, N_TILE, SG in gemm_mpp.metal.
    // 2D dispatch: tgid.x = col tile, tgid.y = row tile.
    const int M_TILE = 64, N_TILE = 32, SG = 4;
    const int grid_x = (N + N_TILE - 1) / N_TILE;
    const int grid_y = (M + M_TILE - 1) / M_TILE;
    MTLSize grid = MTLSizeMake(grid_x, grid_y, 1);
    MTLSize tpg  = MTLSizeMake(32, SG, 1);  // 2D: 32 threads × SG simdgroups

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Convert f16 output back to f32
    uint16_t *pC = (uint16_t *)[bufC contents];
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = f16_to_f32(pC[i]);
}

void metal_gemm_mpp_morton(float *A, float *B, float *C, int M, int N, int K)
{
    static bool initialized = false;
    if (!initialized) { metal_init(); initialized = true; }

    size_t bytesA = (size_t)M * K * sizeof(uint16_t);
    size_t bytesB = (size_t)K * N * sizeof(uint16_t);
    size_t bytesC = (size_t)M * N * sizeof(uint16_t);

    id<MTLBuffer> bufA = [device newBufferWithLength:bytesA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithLength:bytesB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [device newBufferWithLength:bytesC options:MTLResourceStorageModeShared];

    uint16_t *pA = (uint16_t *)[bufA contents];
    uint16_t *pB = (uint16_t *)[bufB contents];
    for (size_t i = 0; i < (size_t)M * K; i++) pA[i] = f32_to_f16(A[i]);
    for (size_t i = 0; i < (size_t)K * N; i++) pB[i] = f32_to_f16(B[i]);

    uint32_t uM = M, uN = N, uK = K;
    id<MTLBuffer> bufM = [device newBufferWithBytes:&uM length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufN = [device newBufferWithBytes:&uN length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [device newBufferWithBytes:&uK length:4 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:mortonPipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufM offset:0 atIndex:3];
    [enc setBuffer:bufN offset:0 atIndex:4];
    [enc setBuffer:bufK offset:0 atIndex:5];

    // Must match M_TILE, N_TILE, SG in gemm_mpp_morton.metal.
    // 1D dispatch — Morton decode in the kernel maps linear index to (col, row).
    const int M_TILE = 64, N_TILE = 32, SG = 4;
    const int grid_x = (N + N_TILE - 1) / N_TILE;
    const int grid_y = (M + M_TILE - 1) / M_TILE;
    MTLSize grid = MTLSizeMake(grid_x * grid_y, 1, 1);
    MTLSize tpg  = MTLSizeMake(SG * 32, 1, 1);

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    uint16_t *pC = (uint16_t *)[bufC contents];
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = f16_to_f32(pC[i]);
}