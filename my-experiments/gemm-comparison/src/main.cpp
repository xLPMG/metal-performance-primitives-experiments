#include <iostream>
#include <vector>
#include <chrono>

extern void cpu_gemm(float *A, float *B, float *C, int N);
// extern void metal_naive_gemm(float *A, float *B, float *C, int N);
// extern void metal_tiled_gemm(float *A, float *B, float *C, int N);

void fill(std::vector<float> &M, float v)
{
    std::fill(M.begin(), M.end(), v);
}

void fill_random(std::vector<float> &M)
{
    for (auto &x : M)
        x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

bool compare(const std::vector<float> &A,
             const std::vector<float> &B,
             float eps = 1e-3f)
{
    for (size_t i = 0; i < A.size(); i++)
    {
        if (std::abs(A[i] - B[i]) > eps)
            return false;
    }
    return true;
}

template <typename F>
long long time_ms(F fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

int main()
{
    int N = 512;

    std::vector<float> A(N * N), B(N * N);
    std::vector<float> C_cpu(N * N), C_naive(N * N), C_tiled(N * N);

    fill_random(A);
    fill_random(B);

    auto cpu_time = time_ms([&]
                            { cpu_gemm(A.data(), B.data(), C_cpu.data(), N); });

    // auto naive_time = time_ms([&]
    //                           { metal_naive_gemm(A.data(), B.data(), C_naive.data(), N); });

    // auto tiled_time = time_ms([&]
    //                           { metal_tiled_gemm(A.data(), B.data(), C_tiled.data(), N); });

    double flops = 2.0 * N * N * N;
    auto gflops = [&](long long ms) { return flops / (ms * 1e6); };

    std::cout << "CPU:   " << cpu_time   << " ms  (" << gflops(cpu_time)   << " GFLOPS)\n";
    // std::cout << "Naive: " << naive_time << " ms  (" << gflops(naive_time) << " GFLOPS)\n";
    // std::cout << "Tiled: " << tiled_time << " ms  (" << gflops(tiled_time) << " GFLOPS)\n";

    // std::cout << "Naive correct: " << compare(C_cpu, C_naive) << "\n";
    // std::cout << "Tiled correct: " << compare(C_cpu, C_tiled) << "\n";
}