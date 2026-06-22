/**
 * @file StaticMLP.h
 * @brief Compile-time-architecture, static-memory multi-layer perceptron.
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * The static-memory counterpart to MLP<T>. Layer sizes, per-layer activation
 * functions, and the loss function are template parameters, so the entire
 * network is fixed at compile time and every buffer is a std::array — there is
 * no heap allocation in construction, inference, or training. Declared as a
 * static/global object, all of its storage lives in .bss/.data.
 *
 * Usage mirrors the dynamic constructor's {sizes}, {activations} ergonomics:
 *
 *   smlp::StaticMLP<float,
 *                   smlp::Layout<2, 16, 8, 8, 1>,
 *                   smlp::Activations<RELU, LINEAR, RELU, SIGMOID>> net;
 *   std::array<float, 2> in{0.1f, 0.2f};
 *   std::array<float, 1> out;
 *   net.GetOutput(in, out);
 *
 * The dynamic MLP<T> is left completely untouched; this type coexists with it
 * and shares the same weight layout and serialisation format for interop.
 *
 * Implemented: construction/init, inference (GetOutput), per-sample Train and
 * mini-batch TrainBatch (RMSProp), the RL surface (SmoothUpdateWeights,
 * CalcGradients, ApplyLoss, AccumulatePolicyGradient), weight-buffer
 * serialisation byte-compatible with MLP<T>, and an inference-only mode
 * (EnableTraining=false) that omits all optimizer state.
 */

#ifndef STATIC_MLP_H
#define STATIC_MLP_H

#include <array>
#include <tuple>
#include <utility>
#include <cstddef>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>

#ifdef ARDUINO
#include <Arduino.h>
#include <SD.h>
#define ENABLE_SAVE_SD 1
#endif

#include "StaticLayer.h"
#include "Loss.h"
#include "utils/Serialise.hpp"

namespace smlp {

// ── Architecture description tags ──
template<std::size_t... Sizes>           struct Layout {};
template<ACTIVATION_FUNCTIONS... Acts>   struct Activations {};

namespace detail {

// Build std::tuple<StaticLayer...> by peeling sizes two-at-a-time (the output
// width of layer i is the input width of layer i+1) in lockstep with the acts.
template<typename T, bool EnableTraining, typename SizesT, typename ActsT>
struct BuildLayers;

template<typename T, bool EnableTraining,
         std::size_t S0, std::size_t S1, std::size_t... Sr,
         ACTIVATION_FUNCTIONS A0, ACTIVATION_FUNCTIONS... Ar>
struct BuildLayers<T, EnableTraining, Layout<S0, S1, Sr...>, Activations<A0, Ar...>> {
    using Head = StaticLayer<T, S0, S1, A0, EnableTraining>;
    using TailTuple =
        typename BuildLayers<T, EnableTraining, Layout<S1, Sr...>, Activations<Ar...>>::type;
    using type = decltype(std::tuple_cat(std::declval<std::tuple<Head>>(),
                                         std::declval<TailTuple>()));
};

template<typename T, bool EnableTraining, std::size_t S0>
struct BuildLayers<T, EnableTraining, Layout<S0>, Activations<>> {
    using type = std::tuple<>;
};

} // namespace detail

// ════════════════════════════════════════════════════════════════════════
//  Loss — compile-time dispatch on raw pointers (no allocation, matches Loss.h)
// ════════════════════════════════════════════════════════════════════════
template<loss::LOSS_FUNCTIONS L, typename T>
inline T compute_loss(const T* expected, const T* actual, T* deriv,
                      std::size_t n, T ssr) {
    if constexpr (L == loss::LOSS_FUNCTIONS::LOSS_MSE) {
        T one_over_n = T(1.0) / static_cast<T>(n);
        T accum = T(0);
        for (std::size_t j = 0; j < n; ++j) {
            T diff = expected[j] - actual[j];
            accum += diff * diff * one_over_n;
            deriv[j] = T(-2.0) * one_over_n * diff * ssr;
        }
        return accum * ssr;
    } else { // LOSS_CATEGORICAL_CROSSENTROPY
        T max_logit = actual[0];
        for (std::size_t i = 1; i < n; ++i) if (actual[i] > max_logit) max_logit = actual[i];
        T sum_exp = T(0);
        for (std::size_t i = 0; i < n; ++i) sum_exp += nn::exp(actual[i] - max_logit);
        T log_sum_exp = max_logit + nn::log(sum_exp);
        T loss = T(0);
        for (std::size_t i = 0; i < n; ++i)
            if (expected[i] > T(0.5)) { loss = -actual[i] + log_sum_exp; break; }
        for (std::size_t i = 0; i < n; ++i) {
            T sm = nn::exp(actual[i] - max_logit) / sum_exp;
            deriv[i] = (sm - expected[i]) * ssr;
        }
        return loss * ssr;
    }
}

// ════════════════════════════════════════════════════════════════════════
//  StaticMLP
// ════════════════════════════════════════════════════════════════════════
template<typename T, typename LayoutT, typename ActsT,
         loss::LOSS_FUNCTIONS Loss = loss::LOSS_FUNCTIONS::LOSS_MSE,
         bool EnableTraining = true>
class StaticMLP;  // primary template

template<typename T, std::size_t... Sizes, ACTIVATION_FUNCTIONS... Acts,
         loss::LOSS_FUNCTIONS Loss, bool EnableTraining>
class StaticMLP<T, Layout<Sizes...>, Activations<Acts...>, Loss, EnableTraining> {
    static_assert(sizeof...(Sizes) >= 2,
                  "StaticMLP needs at least an input and an output layer");
    static_assert(sizeof...(Sizes) == sizeof...(Acts) + 1,
                  "number of activations must equal number of layers (sizes - 1)");

public:
    using value_type   = T;
    using mlp_weights  = std::vector<std::vector<std::vector<T>>>;
    using mlp_biases   = std::vector<std::vector<T>>;
    using training_pair_t = std::pair<std::vector<std::vector<T>>,
                                      std::vector<std::vector<T>>>;

    static constexpr std::size_t kNumLayers = sizeof...(Acts);
    static constexpr std::array<std::size_t, sizeof...(Sizes)> kSizes{ Sizes... };
    static constexpr std::array<ACTIVATION_FUNCTIONS, sizeof...(Acts)> kActs{ Acts... };
    static constexpr std::size_t kNumInputs  = kSizes[0];
    static constexpr std::size_t kNumOutputs = kSizes[sizeof...(Sizes) - 1];

    static constexpr std::size_t compute_max_width() {
        std::size_t m = 0;
        for (std::size_t s : kSizes) if (s > m) m = s;
        return m;
    }
    static constexpr std::size_t kMaxWidth = compute_max_width();

    using LayersTuple =
        typename detail::BuildLayers<T, EnableTraining, Layout<Sizes...>, Activations<Acts...>>::type;

    // ── Storage ──
    LayersTuple m_layers{};                 ///< heterogeneous, fixed-size layers

    // ════════════════════════════════════════════════════════════════════
    //  Construction / init
    // ════════════════════════════════════════════════════════════════════
    StaticMLP() = default;

    /// Fill all weights with a constant (biases left at 0) — mirrors the
    /// dynamic MLP's use_constant_weight_init path.
    void SetConstantWeights(T v) {
        for_each_layer([v](auto & layer) { layer.m_weights.fill(v); });
    }

    /// Seed the internal PRNG used by all randomised init / training shuffle.
    void SetSeed(uint32_t s) { m_rng.seed(s); }

    void InitXavier() {
        for_each_layer([this](auto & layer) { layer.InitXavier(m_rng); });
    }
    void RandomiseWeightsAndBiasesLin(T wmin, T wmax, T bmin, T bmax) {
        for_each_layer([&](auto & layer) { layer.RandomiseLin(m_rng, wmin, wmax, bmin, bmax); });
    }
    void DrawWeights(float scale = 1.f) {
        for_each_layer([&](auto & layer) { layer.DrawWeights(m_rng, scale); });
    }
    void MoveWeights(T speed) {
        for_each_layer([&](auto & layer) { layer.MoveWeights(m_rng, speed); });
    }

    // ── Geometry accessors (parity with MLP<T>) ──
    static constexpr int    get_num_inputs()      { return (int)kNumInputs; }
    static constexpr int    get_num_outputs()     { return (int)kNumOutputs; }
    static constexpr int    get_num_hidden_layers(){ return (int)kNumLayers - 1; }
    static constexpr std::size_t GetNumLayers()   { return kNumLayers; }

    /// Direct access to layer I (replaces dynamic m_layers[i]).
    template<std::size_t I>       auto & layer()       { return std::get<I>(m_layers); }
    template<std::size_t I> const auto & layer() const { return std::get<I>(m_layers); }

    // ── Runtime flat weight access (for jolt-style weight modulation) ──
    // The static layers are heterogeneous tuple elements indexed at compile
    // time; this exposes a flat view over all weights (layer 0 first) so callers
    // can address a weight by a single runtime index.
    static constexpr std::size_t TotalWeights() {
        std::size_t t = 0;
        for (std::size_t i = 0; i < kNumLayers; ++i) t += kSizes[i] * kSizes[i + 1];
        return t;
    }
    /// Pointer to the weight at global flat index `g` (nullptr if out of range).
    T* WeightPtrAt(std::size_t g) { return weight_ptr_impl<0>(g); }

    // ════════════════════════════════════════════════════════════════════
    //  Inference
    // ════════════════════════════════════════════════════════════════════
    /// Core: reads kNumInputs from `input`, writes kNumOutputs to `output`.
    void GetOutput(const T* input, T* output, bool for_inference = true) {
        const T* result = forward_layer<0>(input);
        for (std::size_t i = 0; i < kNumOutputs; ++i) output[i] = result[i];

        if constexpr (Loss == loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY) {
            if (for_inference && kNumOutputs > 1) softmax_inplace(output);
        }
    }

    void GetOutput(const std::array<T, kNumInputs> & input,
                   std::array<T, kNumOutputs> & output,
                   bool for_inference = true) {
        GetOutput(input.data(), output.data(), for_inference);
    }

    /// std::vector overload for source-compatibility with MLP<T> call sites.
    void GetOutput(const std::vector<T> & input,
                   std::vector<T> * output,
                   bool for_inference = true) {
        output->resize(kNumOutputs);
        GetOutput(input.data(), output->data(), for_inference);
    }

    void GetOutputClass(const T* output, std::size_t * class_id) const {
        std::size_t best = 0;
        for (std::size_t i = 1; i < kNumOutputs; ++i)
            if (output[i] > output[best]) best = i;
        *class_id = best;
    }

    // ════════════════════════════════════════════════════════════════════
    //  Training
    // ════════════════════════════════════════════════════════════════════
    /// Per-sample SGD training (matches dynamic MLP<T>::Train). Returns the
    /// final iteration's cost. Samples are processed in order (no shuffle), so
    /// this is deterministic and cross-validates exactly against the dynamic MLP.
    T Train(const training_pair_t & data, float learning_rate,
            int max_iterations = 5000, float min_error_cost = 0.001f) {
        static_assert(EnableTraining, "Train() requires EnableTraining=true");
        const auto & feats = data.first;
        const auto & labels = data.second;
        const T ssr = T(1.0) / static_cast<T>(feats.size());

        T cost = T(0);
        for (int it = 0; it < max_iterations; ++it) {
            cost = T(0);
            for (std::size_t s = 0; s < feats.size(); ++s)
                cost += train_one_sample(feats[s].data(), labels[s].data(),
                                         learning_rate, ssr);
            cost *= ssr;
            if (m_progress_callback) m_progress_callback(it, cost);
            if (cost < nn::from_float<T>(min_error_cost)) break;
        }
        return cost;
    }

    /// Mini-batch RMSProp training (matches dynamic MLP<T>::TrainBatch), with
    /// gradient-norm clipping at 5.0. Uses the internal FastRNG to shuffle.
    T TrainBatch(const training_pair_t & data, float learning_rate,
                 int max_iterations = 5000, std::size_t batch_size = 8,
                 float min_error_cost = 0.001f, bool /*output_log*/ = true) {
        static_assert(EnableTraining, "TrainBatch() requires EnableTraining=true");
        const auto & feats = data.first;
        const auto & labels = data.second;
        const std::size_t n = feats.size();
        const std::size_t n_batches = (n + batch_size - 1) / batch_size;

        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;

        T epoch_loss = T(0);
        for (int it = 0; it < max_iterations; ++it) {
            epoch_loss = T(0);
            // Fisher-Yates shuffle with the internal PRNG.
            for (std::size_t i = n; i > 1; --i)
                std::swap(idx[i - 1], idx[m_rng.next_u32() % i]);

            std::size_t cursor = 0;
            for (std::size_t b = 0; b < n_batches; ++b) {
                std::size_t cur = std::min(batch_size, n - cursor);
                T batch_inv = T(1.0) / static_cast<T>(cur);
                for_each_layer([](auto & l) { l.InitGradientAccumulators(); });

                T batch_loss = T(0);
                for (std::size_t i = 0; i < cur; ++i) {
                    std::size_t s = idx[cursor++];
                    const T* out = forward_layer<0>(feats[s].data());
                    for (std::size_t o = 0; o < kNumOutputs; ++o) m_pred[o] = out[o];
                    batch_loss += compute_loss<Loss>(labels[s].data(), m_pred.data(),
                                                     m_deriv.data(), kNumOutputs, T(1.0));
                    backprop_accumulate<kNumLayers - 1>(m_deriv.data());
                }

                // Gradient-norm clipping (threshold 5.0, matches dynamic).
                T sumsq = T(0);
                for_each_layer([&](auto & l) { sumsq += l.GetGradSumSquared(batch_inv); });
                T norm = nn::sqrt(sumsq);
                if (norm > T(5.0)) {
                    T coef = T(5.0) / norm;
                    for_each_layer([coef](auto & l) { l.ScaleAccumulatedGradients(coef); });
                }
                for_each_layer([&](auto & l) {
                    l.ApplyAccumulatedGradients(learning_rate, batch_inv);
                });
                epoch_loss += batch_loss / static_cast<T>(cur);
            }
            epoch_loss /= static_cast<T>(n_batches);
            if (m_progress_callback) m_progress_callback(it, epoch_loss);
            if (epoch_loss < nn::from_float<T>(min_error_cost)) break;
        }
        return epoch_loss;
    }

    void SetProgressCallback(std::function<void(int, T)> cb) {
        m_progress_callback = std::move(cb);
    }

    // ════════════════════════════════════════════════════════════════════
    //  Reinforcement-learning surface
    // ════════════════════════════════════════════════════════════════════
    /// Polyak soft update of this network's weights toward `src` (target nets).
    /// Same instantiation required — enforced by the type system.
    void SmoothUpdateWeights(const StaticMLP & src, float alpha) {
        const float alphaInv = 1.0f - alpha;
        smooth_update_impl<0>(src, alpha, alphaInv);
    }
    void SmoothUpdateWeights(const std::shared_ptr<StaticMLP> & src, float alpha) {
        SmoothUpdateWeights(*src, alpha);
    }

    /// Back-propagate an externally supplied output-error vector, updating
    /// weights immediately (matches dynamic MLP<T>::ApplyLoss).
    void ApplyLoss(const T* feat, const T* output_error, float learning_rate) {
        static_assert(EnableTraining, "ApplyLoss() requires EnableTraining=true");
        forward_layer<0>(feat);
        for (std::size_t i = 0; i < kNumOutputs; ++i) m_deriv[i] = output_error[i];
        backprop_immediate<kNumLayers - 1>(m_deriv.data(), learning_rate);
    }

    /// Autograd: gradients of the output error w.r.t. inputs, WITHOUT updating
    /// weights (matches dynamic MLP<T>::CalcGradients). Each layer's
    /// input-gradient is left in layer<I>().GetGrads(); if `input_grad_out` is
    /// non-null the network-input gradient (kNumInputs values) is written there.
    void CalcGradients(const T* feat, const T* deriv_error_output,
                       T* input_grad_out = nullptr) {
        static_assert(EnableTraining, "CalcGradients() requires EnableTraining=true");
        forward_layer<0>(feat);
        for (std::size_t i = 0; i < kNumOutputs; ++i) m_deriv[i] = deriv_error_output[i];
        calc_grad_impl<kNumLayers - 1>(m_deriv.data());
        if (input_grad_out) {
            const auto & g = std::get<0>(m_layers).GetGrads();
            for (std::size_t j = 0; j < kNumInputs; ++j) input_grad_out[j] = g[j];
        }
    }

    /// Accumulate a (negated) policy gradient for maximisation
    /// (matches dynamic MLP<T>::AccumulatePolicyGradient).
    void AccumulatePolicyGradient(const T* state, const T* action_gradient) {
        static_assert(EnableTraining, "AccumulatePolicyGradient() requires EnableTraining=true");
        forward_layer<0>(state);
        for (std::size_t i = 0; i < kNumOutputs; ++i) m_deriv[i] = -action_gradient[i];
        backprop_accumulate<kNumLayers - 1>(m_deriv.data());
    }

    // Batch-accumulator controls (parity with dynamic MLP<T>).
    void InitializeAllGradientAccumulators() {
        for_each_layer([](auto & l) { l.InitGradientAccumulators(); });
    }
    void ClearAllGradientAccumulators() { InitializeAllGradientAccumulators(); }
    void ApplyAllAccumulatedGradients(float lr, T batch_size_inv) {
        for_each_layer([&](auto & l) { l.ApplyAccumulatedGradients(lr, batch_size_inv); });
    }

    // ════════════════════════════════════════════════════════════════════
    //  Weight get/set — setup-only convenience (allocates; not a hot path)
    // ════════════════════════════════════════════════════════════════════
    mlp_weights GetAllWeights() const {
        mlp_weights out(kNumLayers);
        get_weights_impl<0>(out);
        return out;
    }

    void SetAllWeights(const mlp_weights & w) { set_weights_impl<0>(w); }

    mlp_biases GetAllBiases() const {
        mlp_biases out(kNumLayers);
        get_biases_impl<0>(out);
        return out;
    }

    void SetAllBiases(const mlp_biases & b) { set_biases_impl<0>(b); }

    // ════════════════════════════════════════════════════════════════════
    //  Serialisation — byte-compatible with dynamic MLP<T> (weights only)
    // ════════════════════════════════════════════════════════════════════
    /// Append this network's weights to `buffer` in the exact layout produced
    /// by MLP<T>::Serialise, so buffers are interchangeable. Returns new head.
    std::size_t Serialise(std::size_t w_head, std::vector<uint8_t> & buffer) const {
        return serialise_impl<0>(static_cast<uint32_t>(w_head), buffer);
    }

    /// Load weights previously written by MLP<T>::Serialise (or this class).
    /// Per-layer shapes are validated against the compile-time architecture;
    /// a mismatched layer is skipped (its weights left unchanged).
    std::size_t FromSerialised(std::size_t r_head, const std::vector<uint8_t> & buffer) {
        return deserialise_impl<0>(static_cast<uint32_t>(r_head), buffer);
    }

    // ════════════════════════════════════════════════════════════════════
    //  SD save / load — byte-compatible with dynamic MLP<T> (weights + biases
    //  + activations), so existing on-card presets load unchanged.
    // ════════════════════════════════════════════════════════════════════
#if defined(ENABLE_SAVE_SD) && ENABLE_SAVE_SD
    bool SaveMLPNetworkToFile(File & file) {
        std::size_t num_inputs = kNumInputs;
        int num_outputs = (int)kNumOutputs;
        int num_hidden  = (int)kNumLayers - 1;
        if (file.write((const char*)&num_inputs, sizeof(num_inputs)) != sizeof(num_inputs)) return false;
        if (file.write((const char*)&num_outputs, sizeof(num_outputs)) != sizeof(num_outputs)) return false;
        if (file.write((const char*)&num_hidden, sizeof(num_hidden)) != sizeof(num_hidden)) return false;
        for (std::size_t i = 0; i < kSizes.size(); ++i) {
            std::size_t node = kSizes[i];
            if (file.write((const char*)&node, sizeof(node)) != sizeof(node)) return false;
        }
        return save_layers_sd<0>(file);
    }

    bool LoadMLPNetworkFromFile(File & file) {
        std::size_t num_inputs = 0;
        int num_outputs = 0, num_hidden = 0;
        if (file.read((uint8_t*)&num_inputs, sizeof(num_inputs)) != sizeof(num_inputs)) return false;
        if (file.read((uint8_t*)&num_outputs, sizeof(num_outputs)) != sizeof(num_outputs)) return false;
        if (file.read((uint8_t*)&num_hidden, sizeof(num_hidden)) != sizeof(num_hidden)) return false;
        if (num_inputs != kNumInputs || num_outputs != (int)kNumOutputs ||
            num_hidden != (int)kNumLayers - 1) return false;
        for (std::size_t i = 0; i < kSizes.size(); ++i) {
            std::size_t node = 0;
            if (file.read((uint8_t*)&node, sizeof(node)) != sizeof(node)) return false;
            if (node != kSizes[i]) return false;
        }
        return load_layers_sd<0>(file);
    }
#endif

    // ── Diagnostics (parity) ──
    bool CheckAndFixWeights() {
        bool any = false;
        for_each_layer([&any](auto & layer) { any |= layer.CheckAndFixWeights(); });
        return any;
    }
    void ResetOptimizerState() {
        for_each_layer([](auto & layer) { layer.ResetOptimizerState(); });
    }
    T GetGlobalWeightNorm() {
        T sumsq = T(0);
        for_each_layer([&sumsq](auto & layer) {
            T n = layer.getWeightNorm(); sumsq += n * n;
        });
        return nn::sqrt(sumsq);
    }

private:
    // ── Forward ping-pong over the layer tuple ──
    std::array<T, kMaxWidth> m_buf_a{};
    std::array<T, kMaxWidth> m_buf_b{};

    // ── Training scratch (sized 0 when training is disabled) ──
    std::array<T, EnableTraining ? kNumOutputs : 0> m_pred{};
    std::array<T, EnableTraining ? kNumOutputs : 0> m_deriv{};
    std::array<T, EnableTraining ? kMaxWidth   : 0> m_bp0{};
    std::array<T, EnableTraining ? kMaxWidth   : 0> m_bp1{};
    FastRNG m_rng{};
    std::function<void(int, T)> m_progress_callback{};

    // One per-sample SGD step: forward (caches inputs), loss, immediate backprop.
    T train_one_sample(const T* feat, const T* label, float lr, T ssr) {
        const T* out = forward_layer<0>(feat);
        for (std::size_t i = 0; i < kNumOutputs; ++i) m_pred[i] = out[i];
        T loss = compute_loss<Loss>(label, m_pred.data(), m_deriv.data(), kNumOutputs, ssr);
        backprop_immediate<kNumLayers - 1>(m_deriv.data(), lr);
        return loss;
    }

    // Descend layers, immediate weight update; err for layer I is the delta
    // produced by layer I+1. Parity of I keeps err and delta on distinct buffers.
    template<std::size_t I>
    void backprop_immediate(const T* err, float lr) {
        auto & layer = std::get<I>(m_layers);
        T* delta = (I & 1) ? m_bp1.data() : m_bp0.data();
        layer.UpdateImmediate(layer.m_cached_input.data(), err, lr, delta);
        if constexpr (I > 0) backprop_immediate<I - 1>(delta, lr);
    }

    template<std::size_t I>
    void backprop_accumulate(const T* err) {
        auto & layer = std::get<I>(m_layers);
        T* delta = (I & 1) ? m_bp1.data() : m_bp0.data();
        layer.AccumulateGradients(layer.m_cached_input.data(), err, delta);
        if constexpr (I > 0) backprop_accumulate<I - 1>(delta);
    }

    template<std::size_t I>
    T* weight_ptr_impl(std::size_t g) {
        auto & L = std::get<I>(m_layers);
        if (g < L.kWeights) return &L.m_weights[g];
        if constexpr (I + 1 < kNumLayers) return weight_ptr_impl<I + 1>(g - L.kWeights);
        else return nullptr;
    }

#if defined(ENABLE_SAVE_SD) && ENABLE_SAVE_SD
    template<std::size_t I>
    bool save_layers_sd(File & file) {
        if (!std::get<I>(m_layers).SaveLayerSD(file)) return false;
        if constexpr (I + 1 < kNumLayers) return save_layers_sd<I + 1>(file);
        else return true;
    }
    template<std::size_t I>
    bool load_layers_sd(File & file) {
        if (!std::get<I>(m_layers).LoadLayerSD(file)) return false;
        if constexpr (I + 1 < kNumLayers) return load_layers_sd<I + 1>(file);
        else return true;
    }
#endif

    template<std::size_t I>
    void calc_grad_impl(const T* err) {
        auto & layer = std::get<I>(m_layers);
        T* delta = (I & 1) ? m_bp1.data() : m_bp0.data();
        layer.CalcGradients(layer.m_cached_input.data(), err, delta);
        if constexpr (I > 0) calc_grad_impl<I - 1>(delta);
    }

    template<std::size_t I>
    void smooth_update_impl(const StaticMLP & src, float alpha, float alphaInv) {
        std::get<I>(m_layers).SmoothUpdateWeights(std::get<I>(src.m_layers), alpha, alphaInv);
        if constexpr (I + 1 < kNumLayers) smooth_update_impl<I + 1>(src, alpha, alphaInv);
    }

    template<std::size_t I>
    const T* forward_layer(const T* in) {
        T* out = (I & 1) ? m_buf_b.data() : m_buf_a.data();
        std::get<I>(m_layers).forward(in, out);
        if constexpr (I + 1 < kNumLayers) return forward_layer<I + 1>(out);
        else                              return out;
    }

    void softmax_inplace(T* p) {
        T total = T(0);
        for (std::size_t i = 0; i < kNumOutputs; ++i) {
            T x = p[i];
            if (x > T(15.0)) x = T(15.0); else if (x < T(-15.0)) x = T(-15.0);
            p[i] = nn::exp(x);
            total += p[i];
        }
        for (std::size_t i = 0; i < kNumOutputs; ++i) p[i] /= total;
    }

    // ── Compile-time tuple iteration helpers ──
    template<typename F, std::size_t I = 0>
    void for_each_layer(F && f) {
        f(std::get<I>(m_layers));
        if constexpr (I + 1 < kNumLayers) for_each_layer<F, I + 1>(std::forward<F>(f));
    }

    template<std::size_t I>
    void get_weights_impl(mlp_weights & out) const {
        const auto & layer = std::get<I>(m_layers);
        out[I].assign(layer.kOut, std::vector<T>(layer.kIn));
        for (std::size_t n = 0; n < layer.kOut; ++n)
            for (std::size_t j = 0; j < layer.kIn; ++j)
                out[I][n][j] = layer.weight(n, j);
        if constexpr (I + 1 < kNumLayers) get_weights_impl<I + 1>(out);
    }

    template<std::size_t I>
    void set_weights_impl(const mlp_weights & w) {
        auto & layer = std::get<I>(m_layers);
        for (std::size_t n = 0; n < layer.kOut; ++n)
            for (std::size_t j = 0; j < layer.kIn; ++j)
                layer.weight(n, j) = w[I][n][j];
        if constexpr (I + 1 < kNumLayers) set_weights_impl<I + 1>(w);
    }

    template<std::size_t I>
    void get_biases_impl(mlp_biases & out) const {
        const auto & layer = std::get<I>(m_layers);
        out[I].assign(layer.kOut, T(0));
        for (std::size_t n = 0; n < layer.kOut; ++n) out[I][n] = layer.bias(n);
        if constexpr (I + 1 < kNumLayers) get_biases_impl<I + 1>(out);
    }

    template<std::size_t I>
    void set_biases_impl(const mlp_biases & b) {
        auto & layer = std::get<I>(m_layers);
        for (std::size_t n = 0; n < layer.kOut; ++n) layer.bias(n) = b[I][n];
        if constexpr (I + 1 < kNumLayers) set_biases_impl<I + 1>(b);
    }

    template<std::size_t I>
    uint32_t serialise_impl(uint32_t head, std::vector<uint8_t> & buf) const {
        const auto & layer = std::get<I>(m_layers);
        std::vector<std::vector<T>> w(layer.kOut, std::vector<T>(layer.kIn));
        for (std::size_t n = 0; n < layer.kOut; ++n)
            for (std::size_t j = 0; j < layer.kIn; ++j)
                w[n][j] = layer.weight(n, j);
        head = Serialise::FromVector2D(head, w, buf);
        if constexpr (I + 1 < kNumLayers) head = serialise_impl<I + 1>(head, buf);
        return head;
    }

    template<std::size_t I>
    uint32_t deserialise_impl(uint32_t head, const std::vector<uint8_t> & buf) {
        auto & layer = std::get<I>(m_layers);
        std::vector<std::vector<T>> w;
        head = Serialise::ToVector2D(head, buf, w);
        if (w.size() == layer.kOut && (layer.kOut == 0 || w[0].size() == layer.kIn)) {
            for (std::size_t n = 0; n < layer.kOut; ++n)
                for (std::size_t j = 0; j < layer.kIn; ++j)
                    layer.weight(n, j) = w[n][j];
        }
        if constexpr (I + 1 < kNumLayers) head = deserialise_impl<I + 1>(head, buf);
        return head;
    }
};

} // namespace smlp

#endif // STATIC_MLP_H
