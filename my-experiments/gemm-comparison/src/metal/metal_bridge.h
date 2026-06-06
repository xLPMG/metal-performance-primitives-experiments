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

#ifdef __cplusplus
}
#endif