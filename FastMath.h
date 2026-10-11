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

/// e^x for x <= 0 only (the log-sum-exp shifted domain). No NaN/sign/overflow
/// handling: branch-free except for one underflow select, rounding via the
/// 1.5*2^23 magic-number trick instead of a float->int->float round trip, FMA
/// reduction, and 2^n applied by an integer add into the exponent field.
/// Returns 0 for x < -86 (e^-86 ~ 4.5e-38 is already below any probability the
/// loss can resolve, and that keeps the exponent add clear of denormals).
/// Precondition: x <= 0 and not NaN; x > 0 is out of contract.
SMLP_CODE_ATTR inline float exp_nonpos_f32(float x) {
    // e^x = 2^(x*log2e) = 2^n * 2^f, n = round(x*log2e), f in [-0.5, 0.5].
    // Two FMAs do the reduction (f = x*log2e - n with the product unrounded);
    // the log2e constant's own rounding error is at most |x|*3e-8, i.e. below
    // 1e-6 for any x that still contributes visibly to a softmax sum.
    constexpr float kMagic = 12582912.0f; // 1.5 * 2^23
    constexpr float kLog2e = 1.44269504f;
    const float t  = __builtin_fmaf(x, kLog2e, kMagic);
    const float fn = t - kMagic;                                  // round(x*log2e), exact
    const int32_t n = static_cast<int32_t>(std::bit_cast<uint32_t>(t)) - 0x4B400000;
    const float f  = __builtin_fmaf(x, kLog2e, -fn);

    // Degree-5 relative-error minimax of 2^f on [-0.5, 0.5] (max rel err 1.1e-7,
    // constant term pinned to 1 so x = 0 is exact), Estrin form:
    //   P = pa + f^2*pb + f^4*pc, pairing adjacent terms.
    const float f2 = f * f;
    const float pa = __builtin_fmaf(6.93147063e-1f, f, 1.0f);
    const float pb = __builtin_fmaf(5.55061512e-2f, f, 2.40222171e-1f);
    const float pc = __builtin_fmaf(1.33022282e-3f, f, 9.67293885e-3f);
    const float y  = __builtin_fmaf(f2 * f2, pc, __builtin_fmaf(f2, pb, pa));

    const float scaled = std::bit_cast<float>(
        static_cast<uint32_t>(static_cast<int32_t>(std::bit_cast<uint32_t>(y)) + (n << 23)));
    return x < -86.0f ? 0.0f : scaled;
}

/// ln(x) for x >= 1 finite (log-sum-exp's sum of shifted exponentials, which
/// lies in [1, n_classes]). x = 2^e * m, m in [1, 2); ln m = 2*atanh(s),
/// s = (m-1)/(m+1) in [0, 1/3], summed to s^11 (error < 1e-7 absolute). One
/// division, no data-dependent branches. Precondition: x >= 1, finite.
SMLP_CODE_ATTR inline float log_ge1_f32(float x) {
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const float e = static_cast<float>(static_cast<int32_t>(bits >> 23) - 127);
    const float m = std::bit_cast<float>((bits & 0x007FFFFFu) | 0x3F800000u);
    const float s  = (m - 1.0f) / (m + 1.0f);
    const float s2 = s * s;
    float q = 1.0f / 11.0f;
    q = __builtin_fmaf(q, s2, 1.0f / 9.0f);
    q = __builtin_fmaf(q, s2, 1.0f / 7.0f);
    q = __builtin_fmaf(q, s2, 1.0f / 5.0f);
    q = __builtin_fmaf(q, s2, 1.0f / 3.0f);
    q = __builtin_fmaf(q, s2, 1.0f);
    return __builtin_fmaf(e, 0.693147181f, 2.0f * s * q);
}

} // namespace smlp

#endif // SMLP_FAST_MATH_H
