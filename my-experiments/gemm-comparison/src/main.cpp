#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

extern "C" void cpu_gemm(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_naive(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_tiled(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_mps(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_mps_f16(float *A, float *B, float *C, int N);
extern "C" void metal_gemm_mpp(float *A, float *B, float *C, int M, int N, int K);
extern "C" void metal_gemm_mpp_morton(float *A, float *B, float *C, int M, int N, int K);

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

bool compare_relative(const std::vector<float> &A, const std::vector<float> &B, float rtol = 1e-2f)
{
    for (size_t i = 0; i < A.size(); i++)
    {
        // Scale by the reference value (B) to avoid false passes when A[i] is near zero
        float scale = std::max(std::abs(B[i]), 1e-6f);
        if (std::abs(A[i] - B[i]) / scale > rtol)
            return false;
    }
    return true;
}

template<typename F>
long long time_us(F fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main()
{
    int N = 2048;

    std::vector<float> A(N * N), B(N * N);
    std::vector<float> C_cpu(N * N), C_naive(N * N), C_tiled(N * N), C_mps(N * N), C_mps_f16(N * N), C_mpp(N * N), C_mpp_morton(N * N);

    fill_random(A);
    fill_random(B);

    // Warm up all Metal backends before timing (avoids JIT/init overhead)
    // metal_gemm_naive(A.data(), B.data(), C_naive.data(), N);
    metal_gemm_tiled(A.data(), B.data(), C_tiled.data(), N);
    metal_gemm_mps(A.data(), B.data(), C_mps.data(), N);
    metal_gemm_mps_f16(A.data(), B.data(), C_mps_f16.data(), N);
    metal_gemm_mpp(A.data(), B.data(), C_mpp.data(), N, N, N);
    metal_gemm_mpp_morton(A.data(), B.data(), C_mpp_morton.data(), N, N, N);

    // auto cpu_time     = time_us([&] { cpu_gemm(A.data(), B.data(), C_cpu.data(), N); });
    auto naive_time   = time_us([&] { metal_gemm_naive(A.data(), B.data(), C_naive.data(), N); });
    auto tiled_time   = time_us([&] { metal_gemm_tiled(A.data(), B.data(), C_tiled.data(), N); });
    auto mps_time     = time_us([&] { metal_gemm_mps(A.data(), B.data(), C_mps.data(), N); });
    auto mps_f16_time = time_us([&] { metal_gemm_mps_f16(A.data(), B.data(), C_mps_f16.data(), N); });
    auto mpp_time        = time_us([&] { metal_gemm_mpp(A.data(), B.data(), C_mpp.data(), N, N, N); });
    auto mpp_morton_time = time_us([&] { metal_gemm_mpp_morton(A.data(), B.data(), C_mpp_morton.data(), N, N, N); });

    double flops = 2.0 * N * N * N;
    auto gflops = [&](long long us) {
        return us > 0 ? flops / (us * 1e3) : 0.0;
    };

    std::cout << "\n";
    std::cout << std::left << std::fixed << std::setprecision(2)
              << std::setw(14) << "Backend"
              << std::setw(12) << "Time (us)"
              << std::setw(16) << "GFLOPS"
              << "Correct\n";
    std::cout << std::string(54, '-') << "\n";

    // std::cout << std::left
    //           << std::setw(10) << "CPU"
    //           << std::setw(12) << cpu_time
    //           << std::setw(16) << gflops(cpu_time)
    //           << "ref\n";

    std::cout << std::left
              << std::setw(14) << "Naive"
              << std::setw(12) << naive_time
              << std::setw(16) << gflops(naive_time)
              << (compare(C_cpu, C_naive) ? "yes" : "no") << "\n";

    std::cout << std::left
              << std::setw(14) << "Tiled"
              << std::setw(12) << tiled_time
              << std::setw(16) << gflops(tiled_time)
              << (compare(C_cpu, C_tiled) ? "yes" : "no") << "\n";

    std::cout << std::left
              << std::setw(14) << "MPS f32"
              << std::setw(12) << mps_time
              << std::setw(16) << gflops(mps_time)
              << (compare(C_cpu, C_mps) ? "yes" : "no") << "\n";

    std::cout << std::left
              << std::setw(14) << "MPS f16"
              << std::setw(12) << mps_f16_time
              << std::setw(16) << gflops(mps_f16_time)
              << (compare_relative(C_cpu, C_mps_f16) ? "yes" : "no") << "\n";

    std::cout << std::left
              << std::setw(14) << "MPP"
              << std::setw(12) << mpp_time
              << std::setw(16) << gflops(mpp_time)
              << (compare_relative(C_cpu, C_mpp) ? "yes" : "no") << "\n";

    std::cout << std::left
              << std::setw(14) << "MPP Morton"
              << std::setw(12) << mpp_morton_time
              << std::setw(16) << gflops(mpp_morton_time)
              << (compare_relative(C_cpu, C_mpp_morton) ? "yes" : "no") << "\n";

    std::cout << "\n";
}