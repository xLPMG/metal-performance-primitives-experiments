// 08_conv2d_basic – depthwise-1 conv2d, half activation, half weights → half output
// NHWC activation, HWIO weights (both required by the current API).
// A minimal 3×3 convolution over a single batch element.
//
// convolution2d_descriptor fields:
//   destination_dimensions: int4(O, dst_W, dst_H, N_batch)  [NHWO order]
//   source_dimensions:      int4(C_in, src_W, src_H, N_batch)
//   kernel_dimensions:      int2(kernel_W, kernel_H)
//   strides:                int2(stride_W, stride_H)
//   dilations:              int2(dilation_W, dilation_H)
//   groups:                 1 (only supported value)
//
// All dimensions must be compile-time constants in the descriptor.
// The kernel is dispatched with one threadgroup per output spatial position.
// execution_simdgroups<1>: single simdgroup per threadgroup for small conv.

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// Fixed-size problem for the descriptor (all values compile-time).
constant int BATCH   = 1;
constant int IN_H    = 8;
constant int IN_W    = 8;
constant int OUT_H   = 6;   // (8 - 3) / 1 + 1
constant int OUT_W   = 6;
constant int C_IN    = 16;
constant int C_OUT   = 32;
constant int KH      = 3;
constant int KW      = 3;

kernel void conv2d_basic(
    device half*  act_ptr     [[buffer(0)]],   // NHWC: [1, H, W, C_IN]
    device half*  weight_ptr  [[buffer(1)]],   // HWIO: [KH, KW, C_IN, C_OUT]
    device half*  dst_ptr     [[buffer(2)]],   // NHWO: [1, OUT_H, OUT_W, C_OUT]
    uint2 tgid [[threadgroup_position_in_grid]])  // (x=out_w, y=out_h)
{
    // Activation: NHWC layout
    auto act = tensor(act_ptr,
                      dextents<int,4>{BATCH, IN_H, IN_W, C_IN},
                      array<int,4>{IN_H*IN_W*C_IN, IN_W*C_IN, C_IN, 1});

    // Weights: HWIO layout (H, W, C_in, C_out)
    auto wgt = tensor(weight_ptr,
                      dextents<int,4>{KH, KW, C_IN, C_OUT},
                      array<int,4>{KW*C_IN*C_OUT, C_IN*C_OUT, C_OUT, 1});

    // Destination: NHWO layout
    auto dst = tensor(dst_ptr,
                      dextents<int,4>{BATCH, OUT_H, OUT_W, C_OUT},
                      array<int,4>{OUT_H*OUT_W*C_OUT, OUT_W*C_OUT, C_OUT, 1});

    constexpr auto desc = convolution2d_descriptor(
        /*dst_dims (O,dstW,dstH,N)*/ int4(C_OUT, OUT_W, OUT_H, BATCH),
        /*src_dims (C,srcW,srcH,N)*/ int4(C_IN,  IN_W,  IN_H,  BATCH),
        /*kernel*/                   int2(KW, KH),
        convolution2d_activation_layout::nhwc,
        convolution2d_weights_layout::hwio,
        /*strides*/   int2(1, 1),
        /*dilations*/ int2(1, 1),
        /*groups*/    1);

    convolution2d<desc, metal::execution_simdgroups<1>> op;
    // set_offsets tells the op which output spatial position this TG handles.
    op.set_offsets(int2((int)tgid.x, (int)tgid.y));

    op.run(act, wgt, dst);
}
