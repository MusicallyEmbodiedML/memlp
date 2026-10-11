/**
 * @file FastMath.h
 * @brief Private float exp/log for the MLP hot path (softmax cross-entropy).
 *
 * Why not std::exp/std::log: on the RP2350 the SDK routes expf/logf through
 * pico_float's `__wrap_expf`/`__wrap_logf`, a single copy shared with core 0's
 * own code (stdio, the RAM flooder), so it lives in core 0's bank. A core-1
 * pinned MLP calling it would silently depend on core 0's bank. These versions
 * are `smlp::`-namespaced and carry SMLP_CODE_ATTR, so they are placed with the
 * rest of the hot path and covered by validate_memory_placement.py's
 * _MLP_SYMBOL_PREFIXES check and its callee-closure check.
 *
 * Algorithms are the classic Cephes single-precision expf/logf (about 1 ulp),
 * using only float add/mul/FMA and integer bit manipulation: no libcalls.
 * Not bit-identical to pico_float's implementation.
 */

#ifndef SMLP_FAST_MATH_H
#define SMLP_FAST_MATH_H

#include <bit>
#include <cstdint>
#include <limits>

#include "Placement.h"

namespace smlp {

/// e^x. Underflows to +0 below ln(FLT_MIN) (no denormal results), overflows
/// to +inf above ln(FLT_MAX); NaN propagates.
SMLP_CODE_ATTR inline float exp_f32(float x) {
    if (!(x == x)) return x;
    if (x < -87.33654f) return 0.0f;
    if (x > 88.72284f) return std::numeric_limits<float>::infinity();

    // x = n*ln2 + r, |r| <= ln2/2, with ln2 split into an exactly
    // representable high part and a small correction.
    const float fn = static_cast<float>(static_cast<int32_t>(x * 1.44269504f + (x < 0.0f ? -0.5f : 0.5f)));
    float r = x - fn * 0.693359375f;
    r -= fn * -2.12194440e-4f;

    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    float y = p * (r * r) + r + 1.0f;

    // 2^n in two factors so n = 128 (x just under ln(FLT_MAX)) doesn't
    // overflow the 8-bit exponent field.
    const int32_t n  = static_cast<int32_t>(fn);
    const int32_t n1 = n / 2;
    const int32_t n2 = n - n1;
    const float s1 = std::bit_cast<float>(static_cast<uint32_t>(n1 + 127) << 23);
    const float s2 = std::bit_cast<float>(static_cast<uint32_t>(n2 + 127) << 23);
    return (y * s1) * s2;
}

/// ln(x). x < 0 gives NaN, x == 0 gives -inf, +inf and NaN propagate.
SMLP_CODE_ATTR inline float log_f32(float x) {
    if (!(x == x)) return x;
    if (x < 0.0f) return std::numeric_limits<float>::quiet_NaN();
    if (x == 0.0f) return -std::numeric_limits<float>::infinity();
    if (x == std::numeric_limits<float>::infinity()) return x;

    int32_t bias_adjust = 0;
    if (x < std::numeric_limits<float>::min()) { // denormal: scale into the normal range
        x *= 8388608.0f; // 2^23
        bias_adjust = -23;
    }
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    int32_t e = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + bias_adjust;
    float m = std::bit_cast<float>((bits & 0x007FFFFFu) | 0x3F800000u); // [1, 2)
    if (m > 1.41421356f) { m *= 0.5f; e += 1; }  // m in [sqrt(.5), sqrt(2))
    const float f = m - 1.0f;
    const float z = f * f;

    float p = 7.0376836292e-2f;
    p = p * f - 1.1514610310e-1f;
    p = p * f + 1.1676998740e-1f;
    p = p * f - 1.2420140846e-1f;
    p = p * f + 1.4249322787e-1f;
    p = p * f - 1.6668057665e-1f;
    p = p * f + 2.0000714765e-1f;
    p = p * f - 2.4999993993e-1f;
    p = p * f + 3.3333331174e-1f;

    const float fe = static_cast<float>(e);
    float y = (f * z) * p;
    y += fe * -2.12194440e-4f;
    y -= 0.5f * z;
    return (f + y) + fe * 0.693359375f;
}

} // namespace smlp

#endif // SMLP_FAST_MATH_H
