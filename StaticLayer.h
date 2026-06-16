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

#include "Utils.h"   // ACTIVATION_FUNCTIONS + utils:: activation math (reused, not modified)

#ifdef ARM_MATH_CM33
#include <arm_math.h>
#endif

namespace smlp {

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

    void seed(uint32_t s) { state_ = s ? s : 0x12345678u; }

    uint32_t next_u32() {
        uint32_t x = state_;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state_ = x;
        return x;
    }

    /// Uniform in [0, 1).
    float next_unit() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }

    /// Uniform in [-r, r].
    float next_sym(float r) { return (next_unit() * 2.0f - 1.0f) * r; }

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
inline T activate(T x) {
    if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)          return utils::sigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::TANH)        return utils::hyperbolic_tan(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::LINEAR)      return utils::linear(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::RELU)        return utils::relu(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSIGMOID) return utils::hardsigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSWISH)   return utils::hardswish(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDTANH)    return utils::hardtanh(x);
    else                                                       return x;
}

template<ACTIVATION_FUNCTIONS A, typename T>
inline T activate_deriv(T x) {
    if constexpr (A == ACTIVATION_FUNCTIONS::SIGMOID)          return utils::deriv_sigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::TANH)        return utils::deriv_hyperbolic_tan(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::LINEAR)      return utils::deriv_linear(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::RELU)        return utils::deriv_relu(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSIGMOID) return utils::deriv_hardsigmoid(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDSWISH)   return utils::deriv_hardswish(x);
    else if constexpr (A == ACTIVATION_FUNCTIONS::HARDTANH)    return utils::deriv_hardtanh(x);
    else                                                       return static_cast<T>(1);
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
    std::array<T, kWeights> m_weights{};       ///< row-major [node*NIn + input]
    std::array<T, NOut>     m_biases{};
    std::array<T, NOut>     m_inner_products{}; ///< cached pre-activation

    // ── Optimizer / backprop state (sized 0 for inference-only builds) ──
    std::array<T, EnableTraining ? kWeights : 0> m_grad_accum{};
    std::array<T, EnableTraining ? NOut    : 0> m_bias_grad_accum{};
    std::array<T, EnableTraining ? kWeights : 0> m_sq_grad_avg{};
    std::array<T, EnableTraining ? NOut    : 0> m_bias_sq_grad_avg{};
    std::array<T, EnableTraining ? NIn     : 0> m_cached_input{};
    std::array<T, EnableTraining ? NIn     : 0> m_grads{}; ///< input-gradient (autograd)

    // ── Accessors (parity with dynamic Layer<T>) ──
    static constexpr int GetInputSize()  { return (int)NIn; }
    static constexpr int GetOutputSize() { return (int)NOut; }

    T&       weight(std::size_t node, std::size_t in)       { return m_weights[node * NIn + in]; }
    const T& weight(std::size_t node, std::size_t in) const { return m_weights[node * NIn + in]; }
    T&       bias(std::size_t node)       { return m_biases[node]; }
    const T& bias(std::size_t node) const { return m_biases[node]; }

    // ── Forward pass ──
    // Reads NIn values from `input`, writes NOut activations to `output`.
    inline void forward(const T* input, T* output) {
        if constexpr (EnableTraining) {
            for (std::size_t j = 0; j < NIn; ++j) m_cached_input[j] = input[j];
        }
        const T* w = m_weights.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T sum = m_biases[i];
#ifdef ARM_MATH_CM33
            float dp;
            arm_dot_prod_f32((const float32_t*)w, (const float32_t*)input,
                             NIn, (float32_t*)&dp);
            sum += dp;
#else
            for (std::size_t j = 0; j < NIn; ++j) sum += w[j] * input[j];
#endif
            m_inner_products[i] = sum;
            output[i] = activate<Act>(sum);
            w += NIn;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    //  Training — RMSProp constants (match dynamic Layer<T>)
    // ════════════════════════════════════════════════════════════════════
    static constexpr float rmsPropDecay    = 0.9f;
    static constexpr float rmsPropDecayInv = 0.1f;
    static constexpr float rmsPropEpsilon  = 1e-6f;

    /// Immediate per-sample SGD update — weights only, biases untouched
    /// (matches Layer<T>::UpdateWeights with accumulate=false). `delta_out`
    /// receives the back-propagated error for the previous layer (size NIn).
    inline void UpdateImmediate(const T* input, const T* deriv_err,
                                float lr, T* delta_out) {
        for (std::size_t j = 0; j < NIn; ++j) delta_out[j] = T(0);
        T* w = m_weights.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T es = deriv_err[i] * activate_deriv<Act>(m_inner_products[i]);
            for (std::size_t j = 0; j < NIn; ++j) {
                delta_out[j] += es * w[j];                       // uses old weight
                w[j] += static_cast<T>(lr * (-(es * input[j]))); // then updates
            }
            w += NIn;
        }
    }

    /// Accumulate gradients for a batch (matches Layer<T>::AccumulateGradients).
    inline void AccumulateGradients(const T* input, const T* deriv_err, T* delta_out) {
        for (std::size_t j = 0; j < NIn; ++j) delta_out[j] = T(0);
        const T* w = m_weights.data();
        T* g = m_grad_accum.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T es = deriv_err[i] * activate_deriv<Act>(m_inner_products[i]);
            for (std::size_t j = 0; j < NIn; ++j) {
                g[j] += input[j] * es;
                delta_out[j] += es * w[j];
            }
            m_bias_grad_accum[i] += es;
            w += NIn;
            g += NIn;
        }
    }

    void InitGradientAccumulators() {
        m_grad_accum.fill(T(0));
        m_bias_grad_accum.fill(T(0));
    }

    /// Apply accumulated gradients with RMSProp (matches Layer<T>::ApplyAccumulatedGradients).
    void ApplyAccumulatedGradients(float lr, T batch_size_inv) {
        const T maxSq = static_cast<T>(1e6);
        const T maxLR = static_cast<T>(1.0);
        const T clip  = static_cast<T>(10.0);

        for (std::size_t k = 0; k < kWeights; ++k) {
            T g = m_grad_accum[k] * batch_size_inv;
            g = (g < -clip) ? -clip : (g > clip ? clip : g);
            m_sq_grad_avg[k] = rmsPropDecay * m_sq_grad_avg[k] + rmsPropDecayInv * g * g;
            if (m_sq_grad_avg[k] > maxSq) m_sq_grad_avg[k] = maxSq;
            T adj = static_cast<T>(lr) / (std::sqrt(m_sq_grad_avg[k]) + static_cast<T>(rmsPropEpsilon));
            if (adj > maxLR) adj = maxLR;
            m_weights[k] -= adj * g;
            m_grad_accum[k] = T(0);
        }
        for (std::size_t i = 0; i < NOut; ++i) {
            T bg = m_bias_grad_accum[i] * batch_size_inv;
            bg = (bg < -clip) ? -clip : (bg > clip ? clip : bg);
            m_bias_sq_grad_avg[i] = rmsPropDecay * m_bias_sq_grad_avg[i] + rmsPropDecayInv * bg * bg;
            if (m_bias_sq_grad_avg[i] > maxSq) m_bias_sq_grad_avg[i] = maxSq;
            T adj = static_cast<T>(lr) / (std::sqrt(m_bias_sq_grad_avg[i]) + static_cast<T>(rmsPropEpsilon));
            if (adj > maxLR) adj = maxLR;
            m_biases[i] -= adj * bg;
            m_bias_grad_accum[i] = T(0);
        }
    }

    float GetGradSumSquared(float batch_size_inv) const {
        T s = T(0);
        for (std::size_t k = 0; k < kWeights; ++k) {
            T scaled = m_grad_accum[k] * batch_size_inv;
            s += scaled * scaled;
        }
        return s;
    }

    void ScaleAccumulatedGradients(T coef) {
        for (std::size_t k = 0; k < kWeights; ++k) m_grad_accum[k] *= coef;
        for (std::size_t i = 0; i < NOut; ++i)     m_bias_grad_accum[i] *= coef;
    }

    /// Propagate error to inputs WITHOUT updating weights (matches
    /// Layer<T>::CalcGradients). Stores the input-gradient in m_grads.
    inline void CalcGradients(const T* /*input*/, const T* deriv_err, T* delta_out) {
        for (std::size_t j = 0; j < NIn; ++j) delta_out[j] = T(0);
        const T* w = m_weights.data();
        for (std::size_t i = 0; i < NOut; ++i) {
            T es = deriv_err[i] * activate_deriv<Act>(m_inner_products[i]);
            for (std::size_t j = 0; j < NIn; ++j) delta_out[j] += es * w[j];
            w += NIn;
        }
        for (std::size_t j = 0; j < NIn; ++j) m_grads[j] = delta_out[j];
    }

    auto &       GetGrads()       { return m_grads; }
    const auto & GetGrads() const { return m_grads; }

    /// Polyak/soft update toward `src` (matches Layer<T>::SmoothUpdateWeights).
    void SmoothUpdateWeights(const StaticLayer & src, float alpha, float alphaInv) {
        for (std::size_t k = 0; k < kWeights; ++k)
            m_weights[k] = alphaInv * m_weights[k] + alpha * src.m_weights[k];
        for (std::size_t i = 0; i < NOut; ++i)
            m_biases[i] = alphaInv * m_biases[i] + alpha * src.m_biases[i];
    }

    // ════════════════════════════════════════════════════════════════════
    //  Initialisation (uses the caller's PRNG)
    // ════════════════════════════════════════════════════════════════════
    /// Xavier/He init (matches Layer<T>::InitXavier limits).
    void InitXavier(FastRNG & rng) {
        float limit;
        if constexpr (Act == ACTIVATION_FUNCTIONS::RELU)
            limit = std::sqrt(6.0f / (float)NIn);                 // He
        else
            limit = std::sqrt(6.0f / (float)(NIn + NOut));        // Xavier
        for (T & w : m_weights) w = static_cast<T>(rng.next_sym(limit));
    }

    void RandomiseLin(FastRNG & rng, T wmin, T wmax, T bmin, T bmax) {
        for (T & w : m_weights) w = static_cast<T>(rng.next_range(wmin, wmax));
        for (T & b : m_biases)  b = static_cast<T>(rng.next_range(bmin, bmax));
    }

    void DrawWeights(FastRNG & rng, float scale) {
        for (T & w : m_weights) w = static_cast<T>(rng.next_sym(scale));
    }

    /// Add Gaussian noise to weights (matches Layer<T>::MoveWeights spirit).
    void MoveWeights(FastRNG & rng, T speed) {
        for (T & w : m_weights) w += static_cast<T>(rng.next_normal((float)speed));
    }

    // ── Diagnostics ──
    T getWeightNorm() const {
        T s = T(0);
        for (const T& w : m_weights) s += w * w;
        return std::sqrt(s);
    }

    bool CheckAndFixWeights() {
        bool corrupt = false;
        for (std::size_t k = 0; k < kWeights; ++k) {
            if (std::isinf(m_weights[k]) || std::isnan(m_weights[k])) {
                m_weights[k] = T(0);
                if constexpr (EnableTraining) m_sq_grad_avg[k] = T(0);
                corrupt = true;
            }
        }
        for (std::size_t i = 0; i < NOut; ++i) {
            if (std::isinf(m_biases[i]) || std::isnan(m_biases[i])) {
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
};

} // namespace smlp

#endif // STATIC_LAYER_H
