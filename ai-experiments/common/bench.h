#pragma once
// GPU benchmark harness.
// Usage: pass a lambda that encodes + commits a MTLCommandBuffer (without waiting).
//        bench() calls waitUntilCompleted and reads GPUStartTime/GPUEndTime.

#import <Metal/Metal.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdio>

struct BenchResult {
    double min_ms, avg_ms, max_ms;

    void print(const char* label, double flops) const {
        double tflops = flops / (min_ms * 1e-3 * 1e12);
        printf("%-44s  min=%7.3f ms  avg=%7.3f ms  %6.2f TFLOPS\n",
               label, min_ms, avg_ms, tflops);
    }
};

// Fn signature: () -> id<MTLCommandBuffer>  (already committed, not yet waited)
template<typename Fn>
BenchResult bench(int n_warmup, int n_iter, Fn&& fn) {
    for (int i = 0; i < n_warmup; i++) {
        id<MTLCommandBuffer> cb = fn();
        [cb waitUntilCompleted];
    }
    std::vector<double> times;
    times.reserve(n_iter);
    for (int i = 0; i < n_iter; i++) {
        id<MTLCommandBuffer> cb = fn();
        [cb waitUntilCompleted];
        double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
        times.push_back(ms);
    }
    double mn  = *std::min_element(times.begin(), times.end());
    double mx  = *std::max_element(times.begin(), times.end());
    double avg = std::accumulate(times.begin(), times.end(), 0.0) / n_iter;
    return {mn, avg, mx};
}
