#pragma once
#include <cstring>
#include <cstdint>

// float32 <-> float16 (half) via Apple clang __fp16 (arm64 host)
static inline uint16_t f32_to_f16(float v) {
    __fp16 h = static_cast<__fp16>(v);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof bits);
    return bits;
}
static inline float f16_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof bits);
    return static_cast<float>(h);
}

// float32 <-> bfloat16 (truncation – same exponent+sign as float32)
static inline uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    // Round to nearest even
    uint32_t rounding_bias = 0x7fff + ((bits >> 16) & 1);
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}
static inline float bf16_to_f32(uint16_t bits) {
    uint32_t expanded = static_cast<uint32_t>(bits) << 16;
    float v;
    std::memcpy(&v, &expanded, 4);
    return v;
}
