#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

extern "C" void cpu_gemm(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_naive(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_tiled(float *A, float *B, float *C, int N);

void fill(std::vector<float> &M, float v)
{
    std::fill(M.begin(), M.end(), v);
}

void fill_random(std::vector<float> &M)
{
    for (auto &x : M)
        x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

bool compare(const std::vector<float> &A, const std::vector<float> &B, float eps = 1e-3f)
{
    for (size_t i = 0; i < A.size(); i++)
    {
        if (std::abs(A[i] - B[i]) > eps)
            return false;
    }
    return true;
}

template<typename F>
long long time_ms(F fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

int main()
{
    int N = 1024;

    std::vector<float> A(N * N), B(N * N);
    std::vector<float> C_cpu(N * N), C_naive(N * N), C_tiled(N * N);

    fill_random(A);
    fill_random(B);

    auto cpu_time = time_ms([&] { cpu_gemm(A.data(), B.data(), C_cpu.data(), N); });

    auto naive_time = time_ms([&] { metal_gemm_naive(A.data(), B.data(), C_naive.data(), N); });

    auto tiled_time = time_ms([&] { metal_gemm_tiled(A.data(), B.data(), C_tiled.data(), N); });

    double flops = 2.0 * N * N * N;
    auto gflops = [&](long long ms) {
        return flops / (ms * 1e6);
    };

    std::cout << "\n";
    std::cout << std::left
              << std::setw(10) << "Backend"
              << std::setw(12) << "Time (ms)"
              << std::setw(14) << "GFLOPS"
              << "Correct\n";
    std::cout << std::string(46, '-') << "\n";

    std::cout << std::left
              << std::setw(10) << "CPU"
              << std::setw(12) << cpu_time
              << std::setw(14) << gflops(cpu_time)
              << "ref\n";

    std::cout << std::left
              << std::setw(10) << "Naive"
              << std::setw(12) << naive_time
              << std::setw(14) << gflops(naive_time)
              << (compare(C_cpu, C_naive) ? "yes" : "no") << "\n";

    std::cout << std::left
              << std::setw(10) << "Tiled"
              << std::setw(12) << tiled_time
              << std::setw(14) << gflops(tiled_time)
              << (compare(C_cpu, C_tiled) ? "yes" : "no") << "\n";

    std::cout << "\n";
}