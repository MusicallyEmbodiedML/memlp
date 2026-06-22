/**
 * @file FixedNN.h
 * @brief Fixed-point support layer that lets StaticMLP/StaticLayer be
 *        instantiated with FixedPoint::Fixed<...> as the value type.
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * The static MLP is already a template on the value type T. This header supplies
 * the three things a fixed-point T needs that a float T gets for free:
 *
 *   1. A type trait `is_fixed_point_v<T>` so the hot paths can switch behaviour
 *      at compile time with `if constexpr` (the float path stays bit-identical).
 *   2. Generic math wrappers in `smlp::nn::` (sqrt/exp/log/isinf/isnan/scale/...)
 *      that forward to <cmath> for float and to fixed-point implementations for
 *      Fixed. exp/log use a float bridge (cold paths only: loss + softmax); sqrt
 *      stays in integer math (used per-weight by RMSProp).
 *   3. Fixed-point activation functions + derivatives in `smlp::fixednn::`,
 *      mirroring utils:: but without std::exp/std::tanh (which have no Fixed
 *      overload). relu/linear/hard* are exact; sigmoid/tanh use a rational
 *      (Padé-style) approximation that saturates to +/-1 and needs no
 *      transcendental — evaluated O(NOut) per layer, never in the NIn MAC loop.
 *
 * IMPORTANT (speed): the inference dot-product MAC is specialised in
 * StaticLayer::forward() to accumulate `w_raw * x_raw` in a 32-bit integer with
 * a single final shift, so Cortex-M0+ uses its single-cycle 32x32->32 multiply
 * and never the (missing, emulated) 64-bit multiply. Keeping the accumulator in
 * range is the caller's job — see the range budget in the project notes:
 * |sum(w*x)| < 2^(31 - 2*FRACTIONAL_BITS).
 */

#ifndef FIXED_NN_H
#define FIXED_NN_H

// ── Host/compat shim ──
// fixedpoint.hpp uses __force_inline (a Pico SDK macro) at namespace scope, so
// it must be defined for the header to even parse off-target (host test runner).
#if !defined(PICO_SDK_VERSION_MAJOR) && !defined(__force_inline)
#define __force_inline inline
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>   // rand() used by Fixed::random()
#include <type_traits>

// Off-target, the Pico HW RNG is absent. Fixed::random_hw()/random_unit_hw()/
// random_bipolar_hw() reference get_rand_32() as a non-dependent name, so GCC's
// two-phase lookup needs a declaration even though we never instantiate them.
#if !defined(PICO_SDK_VERSION_MAJOR)
extern "C" uint32_t get_rand_32();
#endif

#include "fixedpoint.hpp"
#include "Utils.h"   // ACTIVATION_FUNCTIONS

namespace smlp {

// ════════════════════════════════════════════════════════════════════════
//  Trait: is T a FixedPoint::Fixed<...>?
// ════════════════════════════════════════════════════════════════════════
template<typename T> struct is_fixed_point : std::false_type {};
template<int I, int F, typename S>
struct is_fixed_point<FixedPoint::Fixed<I, F, S>> : std::true_type {};
template<typename T>
inline constexpr bool is_fixed_point_v = is_fixed_point<T>::value;

// ════════════════════════════════════════════════════════════════════════
//  nn:: — generic math wrappers (float-transparent, fixed-aware)
// ════════════════════════════════════════════════════════════════════════
namespace nn {

/// sqrt — integer Newton for fixed (no float), std::sqrt for float.
template<typename T> inline T sqrt(T x) {
    if constexpr (is_fixed_point_v<T>) return FixedPoint::sqrt(x);
    else                               return std::sqrt(x);
}

/// exp — float bridge for fixed (cold path: loss + softmax only).
template<typename T> inline T exp(T x) {
    if constexpr (is_fixed_point_v<T>) return T(std::exp(x.to_float()));
    else                               return std::exp(x);
}

/// log — float bridge for fixed (cold path: categorical cross-entropy only).
template<typename T> inline T log(T x) {
    if constexpr (is_fixed_point_v<T>) return T(std::log(x.to_float()));
    else                               return std::log(x);
}

template<typename T> inline bool isinf(T x) {
    if constexpr (is_fixed_point_v<T>) return false;   // fixed-point cannot be inf
    else                               return std::isinf(x);
}
template<typename T> inline bool isnan(T x) {
    if constexpr (is_fixed_point_v<T>) return false;   // ...nor nan
    else                               return std::isnan(x);
}

template<typename T> inline float to_float(T x) {
    if constexpr (is_fixed_point_v<T>) return x.to_float();
    else                               return static_cast<float>(x);
}

/// v * s where s is a runtime float. Float: identical to s*v (commutative).
/// Fixed: convert the scalar once, then multiply.
template<typename T> inline T scale(T v, float s) {
    if constexpr (is_fixed_point_v<T>) return v * T(s);
    else                               return v * s;
}

/// Make a T from a runtime float (e.g. min_error_cost compares).
template<typename T> inline T from_float(float s) {
    if constexpr (is_fixed_point_v<T>) return T(s);
    else                               return static_cast<T>(s);
}

/// Floor a small positive epsilon at 1 LSB for fixed types, so it never
/// underflows to exactly 0 and makes an RMSProp denominator divide-by-zero.
/// Identity for float (keeps that path bit-identical).
template<typename T> inline T floor_eps(T e) {
    if constexpr (is_fixed_point_v<T>) return (e.value == 0) ? T::from_raw(1) : e;
    else                               return e;
}

/// 32-bit "fast" multiply: float -> a*b; fixed -> mul_fast (single 32x32->32
/// product + shift, no int64). Used in the backprop inner loops. The product
/// `a_raw * b_raw` must fit int32 (range budget, same as the forward MAC), which
/// holds for the bounded errors/weights/inputs of a normalised net (F <= 15).
template<typename T> inline T fmul(T a, T b) {
    if constexpr (is_fixed_point_v<T>) return a.mul_fast(b);
    else                               return a * b;
}

/// Reciprocal square root 1/sqrt(x) for a fixed-point x > 0, in 32-bit only
/// (no int64 sqrt/divide). Used by RMSProp as `lr * rsqrt(v)` to replace the
/// per-weight integer sqrt + division. Newton's method on a base-4 normalised
/// mantissa: x = nx * 4^e with nx in [1,4); r0 = 0.5 converges there. Requires
/// 2*FRACTIONAL_BITS <= 30 (true for Q24.7 / Q20.11 / Q17.14).
template<typename T> inline T rsqrt(T x) {
    using S = typename T::storage_type;
    if (x.value <= 0) return T(0);                    // guard: avoid infinite loop
    const S ONE  = T::ONE;
    const S FOUR = ONE << 2;
    S nx = x.value;
    int e = 0;
    while (nx >= FOUR) { nx >>= 2; ++e; }
    while (nx <  ONE)  { nx <<= 2; --e; }
    const T n   = T::from_raw(nx);
    const T c15 = T(1.5f), c05 = T(0.5f);
    T r = T(0.5f);                                   // converges for nx in [1,4)
    for (int i = 0; i < 6; ++i) {
        T r2 = r.mul_fast(r);
        T t  = c15 - c05.mul_fast(n.mul_fast(r2));   // 1.5 - 0.5*nx*r^2
        r = r.mul_fast(t);
    }
    S rr = r.value;                                  // rsqrt(x) = rsqrt(nx) * 2^(-e)
    if (e > 0)      rr >>= e;
    else if (e < 0) rr <<= (-e);
    return T::from_raw(rr);
}

} // namespace nn

// ════════════════════════════════════════════════════════════════════════
//  fixednn:: — fixed-point activations + derivatives (mirror utils::)
// ════════════════════════════════════════════════════════════════════════
namespace fixednn {

// Leaky-ReLU slope (matches utils::kReLUSlope = 0.01).
template<typename T> inline T relu_slope()  { return T(0.01f); }
template<typename T> inline T one_over_six() { return T(0.16666667f); }

template<typename T> inline T linear(T x) { return x; }
template<typename T> inline T deriv_linear(T)  { return T::from_int(1); }

template<typename T> inline T relu(T x) {
    return (x.value > 0) ? x : relu_slope<T>() * x;
}
template<typename T> inline T deriv_relu(T x) {
    return (x.value > 0) ? T::from_int(1) : relu_slope<T>();
}

// tanh via rational approximation: x*(27 + x^2) / (27 + 9 x^2), saturating to
// +/-1 outside |x| >= 3 (the formula already reaches +/-1 there). No transcendental.
template<typename T> inline T tanh(T x) {
    const T three = T::from_int(3);
    if (x > three)  return T::from_int(1);
    if (x < -three) return T::from_int(-1);
    T x2  = x * x;
    T num = x * (T::from_int(27) + x2);
    T den = T::from_int(27) + T::from_int(9) * x2;
    return num / den;
}
template<typename T> inline T deriv_tanh(T x) {
    T t = tanh(x);
    return T::from_int(1) - t * t;
}

// sigmoid(x) = 0.5 * (tanh(x/2) + 1)
template<typename T> inline T sigmoid(T x) {
    T h = tanh(x.div_pow2(1));        // x/2 via shift
    return (h + T::from_int(1)).div_pow2(1);
}
template<typename T> inline T deriv_sigmoid(T x) {
    T s = sigmoid(x);
    return s * (T::from_int(1) - s);
}

template<typename T> inline T hardsigmoid(T x) {
    if (x <= T::from_int(-3)) return T(0);
    if (x >= T::from_int(3))  return T::from_int(1);
    return (x + T::from_int(3)) * one_over_six<T>();
}
template<typename T> inline T deriv_hardsigmoid(T x) {
    return (x > T::from_int(-3) && x < T::from_int(3)) ? one_over_six<T>() : T(0);
}

template<typename T> inline T hardtanh(T x) {
    if (x <= T::from_int(-1)) return T::from_int(-1);
    if (x >= T::from_int(1))  return T::from_int(1);
    return x;
}
template<typename T> inline T deriv_hardtanh(T x) {
    return (x > T::from_int(-1) && x < T::from_int(1)) ? T::from_int(1) : T(0);
}

template<typename T> inline T hardswish(T x) {
    if (x <= T::from_int(-3)) return T(0);
    if (x >= T::from_int(3))  return x;
    return x * (x + T::from_int(3)) * one_over_six<T>();
}
template<typename T> inline T deriv_hardswish(T x) {
    if (x <= T::from_int(-3)) return T(0);
    if (x >= T::from_int(3))  return T::from_int(1);
    return (T::from_int(2) * x + T::from_int(3)) * one_over_six<T>();
}

// Compile-time activation dispatch (mirrors smlp::activate for fixed T).
template<ACTIVATION_FUNCTIONS A, typename T> inline T activate(T x) {
    if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)          return sigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::TANH)        return tanh(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::LINEAR)      return linear(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::RELU)        return relu(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSIGMOID) return hardsigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSWISH)   return hardswish(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDTANH)    return hardtanh(x);
    else                                                       return x;
}
template<ACTIVATION_FUNCTIONS A, typename T> inline T activate_deriv(T x) {
    if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)          return deriv_sigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::TANH)        return deriv_tanh(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::LINEAR)      return deriv_linear(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::RELU)        return deriv_relu(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSIGMOID) return deriv_hardsigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSWISH)   return deriv_hardswish(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDTANH)    return deriv_hardtanh(x);
    else                                                       return T::from_int(1);
}

} // namespace fixednn

} // namespace smlp

#endif // FIXED_NN_H
