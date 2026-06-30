#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

void metal_init();
void metal_shutdown();

void metal_gemm_naive(float *A, float *B, float *C, int N);
void metal_gemm_tiled(float *A, float *B, float *C, int N);
void metal_gemm_mps(float *A, float *B, float *C, int N);
void metal_gemm_mps_f16(float *A, float *B, float *C, int N);
void metal_gemm_mpp(float *A, float *B, float *C, int M, int N, int K);
void metal_gemm_mpp_morton(float *A, float *B, float *C, int M, int N, int K);

#ifdef __cplusplus
}
#endif