# Metal Performance Primitives Experiments

This repository is my workspace for experimenting with Metal Performance Primitives (MPP) on Apple GPUs. The folders ``ai-experiments`` and ``ai-poc`` were created largely by Claude to get me started on my machine (M3 Pro). Real experiments that were actually coded by me for research purposes will be in other directories that are not precented by "ai-". The "gemm-comparison" folder is such an example and contains a simple experiment comparing naive and tiled implementations of matrix multiplication (GEMM) in Metal, which I used to understand the performance benefits of tiling.

This repo might be useful for others who are interested in learning about Metal, MPP, and MPS, as it contains some basic examples.