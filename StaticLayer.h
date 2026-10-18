/**
 * @file StaticLayer.h
 * @brief Compile-time-sized neural-network layer for the static MLP.
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This is the static-memory counterpart to Layer.h. The layer geometry
 * (input width, node count) and activation function are template parameters,
 * so every buffer is a std::array — there is no heap allocation at all. When
 * the owning object is declared static/global the storage lives in .bss/.data.
 *
 * Weight layout matches the dynamic Layer<T> exactly (row-major
 * [num_nodes x num_inputs], weight(node,input) = m_weights[node*NIn + input]),
 * so weights are interchangeable between the dynamic and static networks.
 *
 * Activation selection is resolved at compile time via activate<Act>() — no
 * function pointers, no singleton-map lookup — letting the compiler inline and
 * vectorise the hot dot-product loop.
 *
 * Training-only optimizer state is sized to zero when Train == false, giving an
 * inference build roughly half the per-layer RAM.
 */

#ifndef STATIC_LAYER_H
#define STATIC_LAYER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

#include "Utils.h"   // ACTIVATION_FUNCTIONS + utils:: activation math (reused, not modified)
#include "FixedNN.h" // is_fixed_point_v, nn:: math wrappers, fixednn:: activations
#include "Placement.h" // SMLP_CODE_ATTR hook (blank unless a host project binds it)

#if defined(ARM_MATH_CM33) && defined(__arm__)
#include <arm_math.h>
#endif

namespace smlp {

#if defined(ARM_MATH_CM33) && defined(__arm__)
inline constexpr bool kCmsisDot = true;
#else
inline constexpr bool kCmsisDot = false;
#endif

// ════════════════════════════════════════════════════════════════════════
//  Seedable PRNG — replaces std::random_device/std::mt19937 (no getentropy)
// ════════════════════════════════════════════════════════════════════════
/**
 * @brief Tiny deterministic xorshift32 generator.
 *
 * Avoids std::random_device (which pulls getentropy on bare metal) and the
 * heavyweight std::mt19937 state. Deterministic given a seed.
 */
class FastRNG {
public:
    explicit FastRNG(uint32_t seed = 0x12345678u) : state_(seed ? seed : 0x12345678u) {}

    SMLP_CODE_ATTR void seed(uint32_t s) { state_ = s ? s : 0x12345678u; }

    SMLP_CODE_ATTR uint32_t next_u32() {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }

    /// Uniform in [0, 1).
    SMLP_CODE_ATTR float next_unit() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }

    /// Uniform in [-r, r].
    SMLP_CODE_ATTR float next_sym(float r) { return (next_unit() * 2.0f - 1.0f) * r; }

    /// Uniform in [lo, hi).
    float next_range(float lo, float hi) { return lo + next_unit() * (hi - lo); }

    /// Approximate normal (sum of uniforms), matching utils::gen_randn's spirit.
    float next_normal(float stddev, float mean = 0.f) {
        float accum = next_sym(1.f) + next_sym(1.f) + next_sym(1.f);
        return accum * stddev + mean;
    }

private:
    uint32_t state_;
};

// ════════════════════════════════════════════════════════════════════════
//  Compile-time activation dispatch (reuses the inline math in Utils.h)
// ════════════════════════════════════════════════════════════════════════
template<ACTIVATION_FUNCTIONS A, typename T>
SMLP_CODE_ATTR inline T activate(T x) {
    if constexpr (is_fixed_point_v<T>)                         return fixednn::activate<A>(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)     return utils::sigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::TANH)        return utils::hyperbolic_tan(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::LINEAR)      return utils::linear(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::RELU)        return utils::relu(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSIGMOID) return utils::hardsigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSWISH)   return utils::hardswish(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDTANH)    return utils::hardtanh(x);
    else                                                       return x;
}

template<ACTIVATION_FUNCTIONS A, typename T>
SMLP_CODE_ATTR inline T activate_deriv(T x) {
    if constexpr (is_fixed_point_v<T>)                         return fixednn::activate_deriv<A>(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)     return utils::deriv_sigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::TANH)        return utils::deriv_hyperbolic_tan(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::LINEAR)      return utils::deriv_linear(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::RELU)        return utils::deriv_relu(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSIGMOID) return utils::deriv_hardsigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSWISH)   return utils::deriv_hardswish(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDTANH)    return utils::deriv_hardtanh(x);
    else                                                       return static_cast<T>(1);
}

// Derivative from the cached post-activation output `y` (pre-activation `x` kept
// for the activations whose derivative isn't cleanly a function of y). For fixed
// tanh/sigmoid this avoids re-evaluating the rational approximation (and its
// division) in the backward pass: deriv_tanh = 1 - y^2, deriv_sigmoid = y(1-y).
// Float is left on the exact x-based path so it stays bit-identical to MLP<T>.
template<ACTIVATION_FUNCTIONS A, typename T>
SMLP_CODE_ATTR inline T activate_deriv_cached(T y, T x) {
    if constexpr (is_fixed_point_v<T>) {
        if constexpr (A == ACTIVATION_FUNCTIONS::TANH)
            return T::from_int(1) - y.mul_fast(y);
        else if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)
            return y.mul_fast(T::from_int(1) - y);
        else
            return activate_deriv<A>(x);
    } else {
        return activate_deriv<A>(x);
    }
}

// ════════════════════════════════════════════════════════════════════════
//  StaticLayer
// ════════════════════════════════════════════════════════════════════════
template<typename T, std::size_t NIn, std::size_t NOut,
         ACTIVATION_FUNCTIONS Act, bool EnableTraining = true>
class StaticLayer {
public:
    static constexpr std::size_t kIn      = NIn;
    static constexpr std::size_t kOut     = NOut;
    static constexpr std::size_t kWeights = NIn * NOut;
    static constexpr ACTIVATION_FUNCTIONS kAct = Act;

    // ── Core storage (always present) ──
    // Every streamed array is a PhasedArray (Placement.h): weights, gradient
    // accumulators and squared-gradient averages at SMLP_WEIGHT_PHASE, every
    // activation / scratch stream at SMLP_ACT_PHASE, each from a 16-byte boundary.
    WeightArray<T, kWeights> m_weights{};       ///< row-major [node*NIn + input]
    ActArray<T, NOut>        m_biases{};
    ActArray<T, NOut>        m_inner_products{}; ///< cached pre-activation

    // ── Optimizer / backprop state (sized 0 for inference-only builds) ──
    WeightArray<T, EnableTraining ? kWeights : 0> m_grad_accum{};
    ActArray<T, EnableTraining ? NOut    : 0> m_bias_grad_accum{};
    WeightArray<T, EnableTraining ? kWeights : 0> m_sq_grad_avg{};
    ActArray<T, EnableTraining ? NOut    : 0> m_bias_sq_grad_avg{};
    ActArray<T, EnableTraining ? NIn     : 0> m_cached_input{};
    ActArray<T, EnableTraining ? NIn     : 0> m_grads{}; ///< input-gradient (autograd)
    ActArray<T, EnableTraining ? NOut    : 0> m_act_output{}; ///< cached post-activation y (for cheap derivatives)

    /// Layout check, evaluated at compile time by StaticMLP. Every PhasedArray
    /// member starts on a 16-byte boundary (its data then sits at the phase the
    /// array type was given: SMLP_WEIGHT_PHASE or SMLP_ACT_PHASE), and the layer
    /// is itself 16-byte aligned and a multiple of 16 bytes, so this holds for
    /// any layer position inside an aligned owner.
    static constexpr bool PhasesOk() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
        bool ok = alignof(StaticLayer) == 16 && sizeof(StaticLayer) % 16 == 0
               && __builtin_offsetof(StaticLayer, m_weights)          % 16 == 0
               && __builtin_offsetof(StaticLayer, m_biases)           % 16 == 0
               && __builtin_offsetof(StaticLayer, m_inner_products)   % 16 == 0;
        if constexpr (EnableTraining) {
            ok = ok
               && __builtin_offsetof(StaticLayer, m_grad_accum)       % 16 == 0
               && __builtin_offsetof(StaticLayer, m_sq_grad_avg)      % 16 == 0
               && __builtin_offsetof(StaticLayer, m_bias_grad_accum)  % 16 == 0
               && __builtin_offsetof(StaticLayer, m_bias_sq_grad_avg) % 16 == 0
               && __builtin_offsetof(StaticLayer, m_cached_input)     % 16 == 0
               && __builtin_offsetof(StaticLayer, m_grads)            % 16 == 0
               && __builtin_offsetof(StaticLayer, m_act_output)       % 16 == 0;
        }
#pragma GCC diagnostic pop
        return ok;
    }

    // ── Accessors (parity with dynamic Layer<T>) ──
    static constexpr int GetInputSize()  { return (int)NIn; }
    static constexpr int GetOutputSize() { return (int)NOut; }

    T&       weight(std::size_t node, std::size_t in)       { return m_weights[node * NIn + in]; }
    const T& weight(std::size_t node, std::size_t in) const { return m_weights[node * NIn + in]; }
    T&       bias(std::size_t node)       { return m_biases[node]; }
    const T& bias(std::size_t node) const { return m_biases[node]; }

    // ── Forward pass ──
    // Reads NIn values from `input`, writes NOut activations to `output`.
    //
    // Float path (plan-mlpBetterOptimisation.md Step 3): rows are processed in
    // blocks of 4. Each input value is loaded once and feeds four independent
    // accumulator chains, one per row, each starting at that row's bias and
    // walking j in order (acc += w*x). NOut % 4 is handled by a 2-row and a
    // 1-row block.
    //
    // Multiply and add stay SEPARATE (fp-contract=off, scoped to these two
    // functions): measured on the RP2350 (DWT, SRAM operands), VFMA/VMLA issue
    // every ~3 cycles and cannot overlap, while VMUL + VADD cost ~2.1 per MAC
    // together, so the fused form makes the loop FPU-bound (4.8 vs 3.9
    // cycles/MAC including the loads). The result is therefore a sequential
    // unfused chain per row (golden G1), not the fused chain of 17bc32a.
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
    template<std::size_t R>
    __attribute__((always_inline)) inline void forward_block(const T* __restrict input, T* __restrict output,
                                                              std::size_t i0) {
        const T* __restrict w = m_weights.data() + i0 * NIn;
        T acc[R];
        for (std::size_t r = 0; r < R; ++r) acc[r] = m_biases[i0 + r];
        std::size_t j = 0;
        // unroll 1: no complete peeling (for the 32-input layer GCC otherwise
        // hoists the inputs into spill slots, see Step 1d).
#pragma GCC unroll 1
        for (; j + 2 <= NIn; j += 2) {
            const T x0 = input[j], x1 = input[j + 1];
#pragma GCC unroll 4
            for (std::size_t r = 0; r < R; ++r) {
                acc[r] += w[r * NIn + j]     * x0;
                acc[r] += w[r * NIn + j + 1] * x1;
            }
        }
        if (j < NIn) {
            const T x0 = input[j];
#pragma GCC unroll 4
            for (std::size_t r = 0; r < R; ++r) acc[r] += w[r * NIn + j] * x0;
        }
#pragma GCC unroll 4
        for (std::size_t r = 0; r < R; ++r) {
            m_inner_products[i0 + r] = acc[r];
            output[i0 + r] = activate<Act>(acc[r]);
            if constexpr (EnableTraining) m_act_output[i0 + r] = output[i0 + r];  // cache y for backward
        }
    }

    __attribute__((noinline, noclone)) SMLP_HOT_CODE_ATTR void forward(const T* __restrict input, T* __restrict output) {
        if constexpr (EnableTraining) {
            for (std::size_t j = 0; j < NIn; ++j) m_cached_input[j] = input[j];
        }
        if constexpr (std::is_same_v<T, float> && !kCmsisDot) {
            std::size_t i = 0;
            for (; i + 4 <= NOut; i += 4) forward_block<4>(input, output, i);
            if constexpr (NOut % 4 >= 2) { forward_block<2>(input, output, i); i += 2; }
            if constexpr (NOut % 2 == 1) forward_block<1>(input, output, i);
        } else {
        const T* __restrict w = m_weights.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T sum;
            if constexpr (is_fixed_point_v<T>) {
                // ── Fixed-point MAC: 32-bit only, no int64 (critical on M0+) ──
                // Accumulate raw w*x products (each carries 2*F fractional bits)
                // in the storage integer, add the bias shifted into that space,
                // and do a single arithmetic shift back to F fractional bits.
                // Range budget: |bias + sum(w*x)| < 2^(31 - 2*F).
                using S = typename T::storage_type;
                constexpr int F = T::FRACTIONAL_BITS;
                S acc = static_cast<S>(m_biases[i].value) << F;
                for (std::size_t j = 0; j < NIn; ++j)
                    acc += w[j].value * input[j].value;
                sum = T::from_raw(static_cast<S>(acc >> F));
            } else {
                sum = m_biases[i];
#if defined(ARM_MATH_CM33) && defined(__arm__)
                float dp;
                arm_dot_prod_f32((const float32_t*)w, (const float32_t*)input,
                                 NIn, (float32_t*)&dp);
                sum += dp;
#else
                for (std::size_t j = 0; j < NIn; ++j) sum += w[j] * input[j];
#endif
            }
            m_inner_products[i] = sum;
            output[i] = activate<Act>(sum);
            if constexpr (EnableTraining) m_act_output[i] = output[i];  // cache y for backward
            w += NIn;
        }
        }
    }
#pragma GCC pop_options

    // ════════════════════════════════════════════════════════════════════
    //  Training — RMSProp constants (match dynamic Layer<T>)
    // ════════════════════════════════════════════════════════════════════
    static constexpr float rmsPropDecay    = 0.9f;
    static constexpr float rmsPropDecayInv = 0.1f;
    static constexpr float rmsPropEpsilon  = 1e-6f;

    /// Immediate per-sample SGD update — weights only, biases untouched
    /// (matches Layer<T>::UpdateWeights with accumulate=false). `delta_out`
    /// receives the back-propagated error for the previous layer (size NIn).
    SMLP_CODE_ATTR inline void UpdateImmediate(const T* __restrict input, const T* __restrict deriv_err,
                                float lr, T* __restrict delta_out) {
        for (std::size_t j = 0; j < NIn; ++j) delta_out[j] = T(0);
        T* __restrict w = m_weights.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T es = deriv_err[i] * activate_deriv_cached<Act>(m_act_output[i], m_inner_products[i]);
            if constexpr (is_fixed_point_v<T>) {
                // 32-bit MAC + lr hoisted out of the inner loop (no int64).
                const T es_lr = es.mul_fast(nn::from_float<T>(lr));
                for (std::size_t j = 0; j < NIn; ++j) {
                    delta_out[j] += es.mul_fast(w[j]);           // uses old weight
                    w[j] -= es_lr.mul_fast(input[j]);            // then updates
                }
            } else {
                for (std::size_t j = 0; j < NIn; ++j) {
                    delta_out[j] += es * w[j];                   // uses old weight
                    w[j] += static_cast<T>(lr * (-(es * input[j]))); // then updates
                }
            }
            w += NIn;
        }
    }

    /// One block of R output rows of AccumulateGradients (float path, Step 4).
    /// The R error terms live in registers; per input j: one load of x, R
    /// independent read-modify-write updates of the gradient rows and, with
    /// NeedInputGrad, one load/store of delta[j] covering all R rows. The
    /// delta chain keeps the rows' sequential order (row i0 first), so the
    /// result is bit-identical to the one-row-at-a-time loop (multiply and add
    /// contract to a fused vfma in both).
    template<std::size_t R, bool NeedInputGrad>
    __attribute__((always_inline)) inline void accumulate_block(const T* __restrict input,
                                                                const T* __restrict deriv_err,
                                                                T* __restrict delta_out, std::size_t i0) {
        const T* __restrict w = m_weights.data() + i0 * NIn;
        T* __restrict g = m_grad_accum.data() + i0 * NIn;
        T es[R];
#pragma GCC unroll 4
        for (std::size_t r = 0; r < R; ++r) {
            es[r] = deriv_err[i0 + r] * activate_deriv_cached<Act>(m_act_output[i0 + r], m_inner_products[i0 + r]);
            m_bias_grad_accum[i0 + r] += es[r];
        }
#pragma GCC unroll 1
        for (std::size_t j = 0; j < NIn; ++j) {
            const T x = input[j];
#pragma GCC unroll 4
            for (std::size_t r = 0; r < R; ++r) g[r * NIn + j] += x * es[r];
            if constexpr (NeedInputGrad) {
                T d = delta_out[j];
#pragma GCC unroll 4
                for (std::size_t r = 0; r < R; ++r) d += es[r] * w[r * NIn + j];
                delta_out[j] = d;
            }
        }
    }

    /// Accumulate gradients for a batch (matches Layer<T>::AccumulateGradients).
    /// With NeedInputGrad = false (the network's first layer, whose input
    /// gradient nobody consumes) `delta_out` is never touched and may be nullptr.
    template<bool NeedInputGrad = true>
    __attribute__((noinline, noclone)) SMLP_HOT_CODE_ATTR void AccumulateGradients(const T* __restrict input, const T* __restrict deriv_err, T* __restrict delta_out) {
        if constexpr (NeedInputGrad) {
            for (std::size_t j = 0; j < NIn; ++j) delta_out[j] = T(0);
        }
        if constexpr (std::is_same_v<T, float>) {
            std::size_t i = 0;
            for (; i + 4 <= NOut; i += 4) accumulate_block<4, NeedInputGrad>(input, deriv_err, delta_out, i);
            if constexpr (NOut % 4 >= 2) { accumulate_block<2, NeedInputGrad>(input, deriv_err, delta_out, i); i += 2; }
            if constexpr (NOut % 2 == 1) accumulate_block<1, NeedInputGrad>(input, deriv_err, delta_out, i);
        } else {
        const T* __restrict w = m_weights.data();
        T* __restrict g = m_grad_accum.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T es = deriv_err[i] * activate_deriv_cached<Act>(m_act_output[i], m_inner_products[i]);
            for (std::size_t j = 0; j < NIn; ++j) {
                g[j] += nn::fmul(input[j], es);
                if constexpr (NeedInputGrad) delta_out[j] += nn::fmul(es, w[j]);
            }
            m_bias_grad_accum[i] += es;
            w += NIn;
            g += NIn;
        }
        }
    }

    SMLP_CODE_ATTR void InitGradientAccumulators() {
        m_grad_accum.fill(T(0));
        m_bias_grad_accum.fill(T(0));
    }

    /// Apply accumulated gradients with RMSProp (matches Layer<T>::ApplyAccumulatedGradients).
    __attribute__((noinline, noclone)) SMLP_CODE_ATTR void ApplyAccumulatedGradients(float lr, T batch_size_inv) {
        // maxSq caps the squared-gradient EMA. 1e6 isn't representable in the
        // smaller fixed formats (e.g. Q17.14 maxes ~131072) and would wrap to a
        // negative raw value, so for fixed use the format's largest value — the
        // EMA of clipped gradients (|g|<=clip) never approaches it anyway.
        T maxSq;
        if constexpr (is_fixed_point_v<T>)
            maxSq = T::from_raw(std::numeric_limits<typename T::storage_type>::max());
        else
            maxSq = static_cast<T>(1e6);
        const T maxLR = static_cast<T>(1.0);
        const T clip  = static_cast<T>(10.0);

        for (std::size_t k = 0; k < kWeights; ++k) {
            T g = m_grad_accum[k] * batch_size_inv;
            g = (g < -clip) ? -clip : (g > clip ? clip : g);
            m_sq_grad_avg[k] = nn::scale(m_sq_grad_avg[k], rmsPropDecay)
                             + nn::scale(g, rmsPropDecayInv) * g;
            if (m_sq_grad_avg[k] > maxSq) m_sq_grad_avg[k] = maxSq;
            T adj;
            if constexpr (is_fixed_point_v<T>)   // lr / sqrt(v), 32-bit (no int64 sqrt/divide)
                // v underflows to 0 for tiny gradients (g^2 below the LSB); real
                // RMSProp's lr/(sqrt(v)+eps) -> lr/eps -> clamps to maxLR there, so
                // degrade to bounded SGD rather than a dead (adj=0) update.
                adj = (m_sq_grad_avg[k].value <= 0) ? maxLR
                    : nn::from_float<T>(lr).mul_fast(nn::rsqrt(m_sq_grad_avg[k]));
            else
                adj = static_cast<T>(lr) / (nn::sqrt(m_sq_grad_avg[k]) + static_cast<T>(rmsPropEpsilon));
            if (adj > maxLR) adj = maxLR;
            m_weights[k] -= adj * g;
            m_grad_accum[k] = T(0);
        }
        for (std::size_t i = 0; i < NOut; ++i) {
            T bg = m_bias_grad_accum[i] * batch_size_inv;
            bg = (bg < -clip) ? -clip : (bg > clip ? clip : bg);
            m_bias_sq_grad_avg[i] = nn::scale(m_bias_sq_grad_avg[i], rmsPropDecay)
                                  + nn::scale(bg, rmsPropDecayInv) * bg;
            if (m_bias_sq_grad_avg[i] > maxSq) m_bias_sq_grad_avg[i] = maxSq;
            T adj;
            if constexpr (is_fixed_point_v<T>)   // lr / sqrt(v), 32-bit (no int64 sqrt/divide)
                adj = (m_bias_sq_grad_avg[i].value <= 0) ? maxLR
                    : nn::from_float<T>(lr).mul_fast(nn::rsqrt(m_bias_sq_grad_avg[i]));
            else
                adj = static_cast<T>(lr) / (nn::sqrt(m_bias_sq_grad_avg[i]) + static_cast<T>(rmsPropEpsilon));
            if (adj > maxLR) adj = maxLR;
            m_biases[i] -= adj * bg;
            m_bias_grad_accum[i] = T(0);
        }
    }

    SMLP_CODE_ATTR T GetGradSumSquared(T batch_size_inv) const {
        T s = T(0);
        for (std::size_t k = 0; k < kWeights; ++k) {
            T scaled = m_grad_accum[k] * batch_size_inv;
            s += scaled * scaled;
        }
        return s;
    }

    SMLP_CODE_ATTR void ScaleAccumulatedGradients(T coef) {
        for (std::size_t k = 0; k < kWeights; ++k) m_grad_accum[k] *= coef;
        for (std::size_t i = 0; i < NOut; ++i)     m_bias_grad_accum[i] *= coef;
    }

    /// Propagate error to inputs WITHOUT updating weights (matches
    /// Layer<T>::CalcGradients). Stores the input-gradient in m_grads.
    inline void CalcGradients(const T* __restrict /*input*/, const T* __restrict deriv_err, T* __restrict delta_out) {
        for (std::size_t j = 0; j < NIn; ++j) delta_out[j] = T(0);
        const T* __restrict w = m_weights.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T es = deriv_err[i] * activate_deriv_cached<Act>(m_act_output[i], m_inner_products[i]);
            for (std::size_t j = 0; j < NIn; ++j) delta_out[j] += nn::fmul(es, w[j]);
            w += NIn;
        }
        for (std::size_t j = 0; j < NIn; ++j) m_grads[j] = delta_out[j];
    }

    auto &       GetGrads()       { return m_grads; }
    const auto & GetGrads() const { return m_grads; }

    /// Polyak/soft update toward `src` (matches Layer<T>::SmoothUpdateWeights).
    void SmoothUpdateWeights(const StaticLayer & src, float alpha, float alphaInv) {
        for (std::size_t k = 0; k < kWeights; ++k)
            m_weights[k] = nn::scale(m_weights[k], alphaInv) + nn::scale(src.m_weights[k], alpha);
        for (std::size_t i = 0; i < NOut; ++i)
            m_biases[i] = nn::scale(m_biases[i], alphaInv) + nn::scale(src.m_biases[i], alpha);
    }

    // ════════════════════════════════════════════════════════════════════
    //  Initialisation (uses the caller's PRNG)
    // ════════════════════════════════════════════════════════════════════
    /// Xavier/He init (matches Layer<T>::InitXavier limits). Biases reset to
    /// zero: InitXavier() only randomizes weights on a fresh instance (where
    /// m_biases already defaults to zero), so a reused instance must be
    /// reset explicitly to reproduce that same starting state.
    SMLP_CODE_ATTR void InitXavier(FastRNG & rng) {
        float limit;
        if constexpr (Act == ACTIVATION_FUNCTIONS::RELU)
            limit = std::sqrt(6.0f / (float)NIn);                 // He
        else
            limit = std::sqrt(6.0f / (float)(NIn + NOut));        // Xavier
        for (T & w : m_weights) w = static_cast<T>(rng.next_sym(limit));
        m_biases.fill(T(0));
    }

    void RandomiseLin(FastRNG & rng, T wmin, T wmax, T bmin, T bmax) {
        for (T & w : m_weights)
            w = static_cast<T>(rng.next_range(nn::to_float(wmin), nn::to_float(wmax)));
        for (T & b : m_biases)
            b = static_cast<T>(rng.next_range(nn::to_float(bmin), nn::to_float(bmax)));
    }

    void DrawWeights(FastRNG & rng, float scale) {
        for (T & w : m_weights) w = static_cast<T>(rng.next_sym(scale));
    }

    /// Add Gaussian noise to weights (matches Layer<T>::MoveWeights spirit).
    void MoveWeights(FastRNG & rng, T speed) {
        for (T & w : m_weights) w += static_cast<T>(rng.next_normal(nn::to_float(speed)));
    }

    // ── Diagnostics ──
    T getWeightNorm() const {
        T s = T(0);
        for (const T& w : m_weights) s += w * w;
        return nn::sqrt(s);
    }

    bool CheckAndFixWeights() {
        bool corrupt = false;
        for (std::size_t k = 0; k < kWeights; ++k) {
            if (nn::isinf(m_weights[k]) || nn::isnan(m_weights[k])) {
                m_weights[k] = T(0);
                if constexpr (EnableTraining) m_sq_grad_avg[k] = T(0);
                corrupt = true;
            }
        }
        for (std::size_t i = 0; i < NOut; ++i) {
            if (nn::isinf(m_biases[i]) || nn::isnan(m_biases[i])) {
                m_biases[i] = T(0);
                if constexpr (EnableTraining) m_bias_sq_grad_avg[i] = T(0);
                corrupt = true;
            }
        }
        return corrupt;
    }

    void ResetOptimizerState() {
        if constexpr (EnableTraining) {
            m_sq_grad_avg.fill(T(0));
            m_bias_sq_grad_avg.fill(T(0));
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  SD save / load — byte-compatible with dynamic Layer<T>::SaveLayerSD
    //  (so models saved by the dynamic library still load here, and vice versa)
    // ════════════════════════════════════════════════════════════════════
#if defined(ENABLE_SAVE_SD) && ENABLE_SAVE_SD
    bool SaveLayerSD(File &file) const {
        std::size_t num_nodes  = NOut;
        std::size_t num_inputs = NIn;
        ACTIVATION_FUNCTIONS act = Act;
        if (file.write((char*)&num_nodes, sizeof(num_nodes)) != sizeof(num_nodes)) return false;
        if (file.write((char*)&num_inputs, sizeof(num_inputs)) != sizeof(num_inputs)) return false;
        if (file.write((char*)&act, sizeof(act)) != sizeof(act)) return false;

        std::size_t n_inputs = NIn;
        for (std::size_t i = 0; i < NOut; ++i) {
            if (file.write((char*)&n_inputs, sizeof(n_inputs)) != sizeof(n_inputs)) return false;
            if (file.write((char*)&m_biases[i], sizeof(T)) != sizeof(T)) return false;
            const T *row = m_weights.data() + i * NIn;
            int dataSize = NIn * sizeof(T);
            if (file.write((char*)row, dataSize) != dataSize) return false;
        }
        return true;
    }

    // Reads the dynamic-format layer and validates it matches this compile-time
    // geometry/activation; returns false on any mismatch (caller falls back).
    bool LoadLayerSD(File &file) {
        std::size_t num_nodes = 0, num_inputs = 0;
        ACTIVATION_FUNCTIONS act{};
        if (file.read((uint8_t*)&num_nodes, sizeof(num_nodes)) != sizeof(num_nodes)) return false;
        if (file.read((uint8_t*)&num_inputs, sizeof(num_inputs)) != sizeof(num_inputs)) return false;
        if (file.read((uint8_t*)&act, sizeof(act)) != sizeof(act)) return false;
        if (num_nodes != NOut || num_inputs != NIn || act != Act) return false;

        for (std::size_t i = 0; i < NOut; ++i) {
            std::size_t n_inputs = 0;
            if (file.read((uint8_t*)&n_inputs, sizeof(n_inputs)) != sizeof(n_inputs)) return false;
            if (file.read((uint8_t*)&m_biases[i], sizeof(T)) != sizeof(T)) return false;
            T *row = m_weights.data() + i * NIn;
            int dataSize = NIn * sizeof(T);
            if (file.read((uint8_t*)row, dataSize) != dataSize) return false;
        }
        return true;
    }
#endif
};

} // namespace smlp

#endif // STATIC_LAYER_H
