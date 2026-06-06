#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <string>

static uint16_t f32_to_f16(float f)
{
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

struct Config
{
    char name[32];
    int M_tile, N_tile, simdgroups;
};

static std::vector<Config> build_configs()
{
    static const struct { int m, n; } TILES[] = {
        {32,16}, {32,32}, {64,32}, {64,64}, {128,32}, {128,64}, {128,128}
    };
    std::vector<Config> cfgs;
    for (auto &t : TILES)
        for (int sg = 1; sg <= 32; sg++) {
            Config c;
            snprintf(c.name, sizeof(c.name), "%dx%d_sg%d", t.m, t.n, sg);
            c.M_tile = t.m; c.N_tile = t.n; c.simdgroups = sg;
            cfgs.push_back(c);
        }
    return cfgs;
}

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;

static void dispatch_gemm(
    id<MTLComputePipelineState> pso,
    id<MTLBuffer> bufA,
    id<MTLBuffer> bufB,
    id<MTLBuffer> bufC,
    id<MTLBuffer> bufM,
    id<MTLBuffer> bufN,
    id<MTLBuffer> bufK,
    int mat_N,
    const Config &cfg)
{
    id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufM offset:0 atIndex:3];
    [enc setBuffer:bufN offset:0 atIndex:4];
    [enc setBuffer:bufK offset:0 atIndex:5];

    // One threadgroup per output tile; threads = simdgroups × 32
    MTLSize grid = MTLSizeMake((mat_N + cfg.N_tile - 1) / cfg.N_tile, (mat_N + cfg.M_tile - 1) / cfg.M_tile, 1);
    MTLSize tpg = MTLSizeMake(32, cfg.simdgroups, 1);

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

int main()
{
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue = [gDevice newCommandQueue];

    auto configs = build_configs();
    const int N_CONFIGS = (int)configs.size(); // 224

    // N values to sweep — all are multiples of 128 (largest tile)
    static const int N_VALUES[] = {512, 1024, 2048, 4096};
    static const int N_N_VALUES = 4;

    // Allocate max-size buffers once (4096×4096 × f16 = 32 MB each)
    const int MAX_N = 4096;
    size_t max_sz = (size_t)MAX_N * MAX_N;

    srand(42);
    std::vector<uint16_t> hA(max_sz), hB(max_sz);
    for (auto &x : hA) x = f32_to_f16((float)rand() / static_cast<float>(RAND_MAX));
    for (auto &x : hB) x = f32_to_f16((float)rand() / static_cast<float>(RAND_MAX));

    id<MTLBuffer> bufA = [gDevice newBufferWithBytes:hA.data() length:max_sz * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [gDevice newBufferWithBytes:hB.data() length:max_sz * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [gDevice newBufferWithLength:max_sz * 2 options:MTLResourceStorageModeShared];

    uint32_t uDim = 0;
    id<MTLBuffer> bufM = [gDevice newBufferWithBytes:&uDim length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufN = [gDevice newBufferWithBytes:&uDim length:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [gDevice newBufferWithBytes:&uDim length:4 options:MTLResourceStorageModeShared];

    // Find reference config (64x32_sg4) and load its PSO once
    int ref_idx = -1;
    for (int i = 0; i < N_CONFIGS; i++)
        if (strcmp(configs[i].name, "64x32_sg4") == 0) { ref_idx = i; break; }

    NSString *ref_path = [NSString stringWithFormat:@"build/%s.metallib", configs[ref_idx].name];
    if (![[NSFileManager defaultManager] fileExistsAtPath:ref_path]) {
        fprintf(stderr, "Error: reference metallib %s not found. Run 'make' first.\n", [ref_path UTF8String]);
        return 1;
    }
    NSError *err = nil;
    id<MTLLibrary> ref_lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:ref_path] error:&err];
    id<MTLFunction> ref_fn = [ref_lib newFunctionWithName:@"gemm_mpp"];
    id<MTLComputePipelineState> ref_pso = [gDevice newComputePipelineStateWithFunction:ref_fn error:&err];

    const char *csv_path = "results.csv";
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { fprintf(stderr, "Could not open %s for writing\n", csv_path); return 1; }
    fprintf(csv, "config,m_tile,n_tile,simdgroups,mat_n,correct,time_us,gflops\n");

    for (int ni = 0; ni < N_N_VALUES; ni++)
    {
        const int N = N_VALUES[ni];
        size_t sz = (size_t)N * N;
        double flops = 2.0 * N * N * N;

        // Update dimension buffers
        uint32_t uN = (uint32_t)N;
        memcpy([bufM contents], &uN, 4);
        memcpy([bufN contents], &uN, 4);
        memcpy([bufK contents], &uN, 4);

        // Compute reference output for this N
        printf("Computing reference for N=%d...\n", N); fflush(stdout);
        dispatch_gemm(ref_pso, bufA, bufB, bufC, bufM, bufN, bufK, N, configs[ref_idx]);
        std::vector<uint16_t> ref_out(sz);
        memcpy(ref_out.data(), [bufC contents], sz * 2);

        printf("Benchmarking N=%d (%d configs)...\n", N, N_CONFIGS); fflush(stdout);

        for (int i = 0; i < N_CONFIGS; i++)
        {
            const Config &cfg = configs[i];
            NSString *path = [NSString stringWithFormat:@"build/%s.metallib", cfg.name];

            if (![[NSFileManager defaultManager] fileExistsAtPath:path])
            {
                fprintf(csv, "%s,%d,%d,%d,%d,rejected,,\n", cfg.name, cfg.M_tile, cfg.N_tile, cfg.simdgroups, N);
                continue;
            }

            NSError *e = nil;
            id<MTLLibrary> lib = [gDevice newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
            id<MTLFunction> fn = lib ? [lib newFunctionWithName:@"gemm_mpp"] : nil;
            id<MTLComputePipelineState> pso = fn ? [gDevice newComputePipelineStateWithFunction:fn error:&e] : nil;

            if (!pso)
            {
                fprintf(csv, "%s,%d,%d,%d,%d,load_error,,\n", cfg.name, cfg.M_tile, cfg.N_tile, cfg.simdgroups, N);
                continue;
            }

            // Warmup
            dispatch_gemm(pso, bufA, bufB, bufC, bufM, bufN, bufK, N, cfg);

            // Timed: average over 10 runs
            const int RUNS = 10;
            long long total_us = 0;
            for (int r = 0; r < RUNS; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                dispatch_gemm(pso, bufA, bufB, bufC, bufM, bufN, bufK, N, cfg);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            }
            long long us = total_us / RUNS;

            // Correctness vs reference
            uint16_t *pC = (uint16_t *)[bufC contents];
            bool ok = true;
            for (size_t j = 0; j < sz && ok; j++) {
                float got = f16_to_f32(pC[j]);
                float exp = f16_to_f32(ref_out[j]);
                float scale = std::max(std::abs(exp), 1e-4f);
                if (std::abs(got - exp) / scale > 1e-2f)
                    ok = false;
            }

            double gflops = us > 0 ? flops / (us * 1e3) : 0.0;
            fprintf(csv, "%s,%d,%d,%d,%d,%s,%lld,%.2f\n",
                    cfg.name, cfg.M_tile, cfg.N_tile, cfg.simdgroups, N,
                    ok ? "yes" : "no", us, gflops);
        }
        printf("  done.\n"); fflush(stdout);
    }

    fclose(csv);
    printf("\nResults written to %s\n\n", csv_path);
    return 0;
}
