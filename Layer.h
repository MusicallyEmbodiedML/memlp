/**
 * @file Layer.h
 * @brief Neural network layer with flat row-major weight matrix for cache-friendly access
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * Weight layout: row-major matrix of size [num_nodes x num_inputs]
 *   weight(node_i, input_j) = m_weights[node_i * num_inputs + input_j]
 *
 * All per-node data (biases, optimizer state) stored in contiguous arrays indexed by node.
 * This replaces the previous per-Node heap allocation, enabling:
 *   - Linear memory access in forward/backward passes
 *   - CMSIS-DSP arm_mat_vec_mult_f32 for the entire layer
 *   - Compiler auto-vectorisation of inner loops
 */

#ifndef LAYER_H
#define LAYER_H

#ifdef ARDUINO

#include <Arduino.h>
#include <VFS.h>
#include <LittleFS.h>
#define ENABLE_SAVE    1

#endif

#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <span>

#include "Utils.h"

#ifdef ARM_MATH_CM33
#include <arm_math.h>
#endif

/**
 * @brief Definition of activation function pointer
 */
template<typename T>
using activation_func_t = T(*)(T);


template<typename T>
class Layer {
public:

  // ════════════════════════════════════════════════════════════════════
  //  Construction
  // ════════════════════════════════════════════════════════════════════

  Layer() = default;

  Layer(int num_inputs_per_node,
        int num_nodes,
        const ACTIVATION_FUNCTIONS & activation_function,
        bool use_constant_weight_init = true,
        T constant_weight_init = 0.5)
      : m_num_inputs_per_node(num_inputs_per_node)
      , m_num_nodes(num_nodes)
  {
    // Resolve activation functions
    std::pair<activation_func_t<T>, activation_func_t<T>> *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
        GetActivationFunctionPair(activation_function, &pair);
    assert(ret_val);
    m_activation_function = pair->first;
    m_deriv_activation_function = pair->second;
    m_activation_function_type = activation_function;

    const size_t total_weights = (size_t)num_nodes * (size_t)num_inputs_per_node;

    // ── Core storage ──
    m_biases.resize(num_nodes, T(0));
    m_inner_products.resize(num_nodes);

    if (use_constant_weight_init) {
      m_weights.resize(total_weights, constant_weight_init);
    } else {
      m_weights.resize(total_weights);
      std::generate(m_weights.begin(), m_weights.end(), utils::gen_rand<T>());
    }

    // ── Optimizer state ──
    m_grad_accum.resize(total_weights, T(0));
    m_bias_grad_accum.resize(num_nodes, T(0));
    m_sq_grad_avg.resize(total_weights, T(0));
    m_bias_sq_grad_avg.resize(num_nodes, T(0));
  }

  ~Layer() = default;


  // ════════════════════════════════════════════════════════════════════
  //  Accessors
  // ════════════════════════════════════════════════════════════════════

  int GetInputSize()  const { return m_num_inputs_per_node; }
  int GetOutputSize() const { return m_num_nodes; }

  /// Direct weight access: weight(node_i, input_j)
  T& weight(size_t node, size_t input) {
    return m_weights[node * m_num_inputs_per_node + input];
  }
  const T& weight(size_t node, size_t input) const {
    return m_weights[node * m_num_inputs_per_node + input];
  }

  T& bias(size_t node)             { return m_biases[node]; }
  const T& bias(size_t node) const { return m_biases[node]; }

  /// Get weights for a single node as a span (for compatibility)
  std::span<T> GetNodeWeights(size_t node) {
    return { m_weights.data() + node * m_num_inputs_per_node,
             (size_t)m_num_inputs_per_node };
  }
  std::span<const T> GetNodeWeights(size_t node) const {
    return { m_weights.data() + node * m_num_inputs_per_node,
             (size_t)m_num_inputs_per_node };
  }

  /// Get all weights as flat span
  std::span<T> GetWeightsFlat() { return m_weights; }
  std::span<const T> GetWeightsFlat() const { return m_weights; }

  /// Get all biases
  std::span<T> GetBiases() { return m_biases; }
  std::span<const T> GetBiases() const { return m_biases; }


  // ════════════════════════════════════════════════════════════════════
  //  Output caching
  // ════════════════════════════════════════════════════════════════════

  void SetCachedOutputs(bool onOrOff) {
    m_cacheOutputs = onOrOff;
    if (!m_cacheOutputs) cachedOutputs.clear();
  }


  // ════════════════════════════════════════════════════════════════════
  //  Forward pass
  // ════════════════════════════════════════════════════════════════════

  inline void GetOutputAfterActivationFunction(const std::vector<T> &input,
                                               std::vector<T> *output) {
    assert(input.size() == m_num_inputs_per_node);
    output->resize(m_num_nodes);

    #ifdef ARM_MATH_CM33
    // ── CMSIS-DSP: per-row dot product over contiguous weight matrix ──
    const T *w_ptr = m_weights.data();
    for (size_t i = 0; i < m_num_nodes; ++i) {
      T sum;
      arm_dot_prod_f32(
          (const float32_t*)w_ptr,
          (const float32_t*)input.data(),
          m_num_inputs_per_node,
          (float32_t*)&sum);
      sum += m_biases[i];
      m_inner_products[i] = sum;
      (*output)[i] = m_activation_function(sum);
      w_ptr += m_num_inputs_per_node;
    }
    #else
    // ── Scalar fallback: still contiguous, auto-vectorisable ──
    const T *w_ptr = m_weights.data();
    for (size_t i = 0; i < m_num_nodes; ++i) {
      T sum = m_biases[i];
      for (size_t j = 0; j < m_num_inputs_per_node; ++j) {
        sum += w_ptr[j] * input[j];
      }
      m_inner_products[i] = sum;
      (*output)[i] = m_activation_function(sum);
      w_ptr += m_num_inputs_per_node;
    }
    #endif

    if (m_cacheOutputs) {
      cachedOutputs = *output;
    }
  }


  // ════════════════════════════════════════════════════════════════════
  //  Gradient accumulation (batch training)
  // ════════════════════════════════════════════════════════════════════

  void InitializeGradientAccumulators() {
    std::fill(m_grad_accum.begin(), m_grad_accum.end(), T(0));
    std::fill(m_bias_grad_accum.begin(), m_bias_grad_accum.end(), T(0));
  }

  void ClearGradientAccumulators() {
    InitializeGradientAccumulators();
  }

  void AccumulateGradients(const std::vector<T>& input,
                           const std::vector<T>& deriv_error,
                           std::vector<T>* deltas) {
    assert(input.size() == m_num_inputs_per_node);
    assert(deriv_error.size() == m_num_nodes);
    deltas->assign(m_num_inputs_per_node, T(0));

    const T *w_ptr = m_weights.data();
    T *g_ptr = m_grad_accum.data();

    for (size_t i = 0; i < m_num_nodes; ++i) {
      T error_signal = deriv_error[i] *
                       m_deriv_activation_function(m_inner_products[i]);

      for (size_t j = 0; j < m_num_inputs_per_node; ++j) {
        g_ptr[j] += input[j] * error_signal;
        (*deltas)[j] += error_signal * w_ptr[j];
      }
      m_bias_grad_accum[i] += error_signal;

      w_ptr += m_num_inputs_per_node;
      g_ptr += m_num_inputs_per_node;
    }
  }


  // ════════════════════════════════════════════════════════════════════
  //  Apply gradients (RMSProp)
  // ════════════════════════════════════════════════════════════════════

  static constexpr float rmsPropDecay    = 0.9f;
  static constexpr float rmsPropDecayInv = 0.1f;
  static constexpr float rmsPropEpsilon  = 1e-6f;

  void ApplyAccumulatedGradients(float learning_rate, T batch_size_inv) {
    const T maxSquaredGradAvg = static_cast<T>(1e6);
    const T maxAdjustedLR     = static_cast<T>(1.0);
    const T gradientClipValue = static_cast<T>(10.0);

    const size_t total_w = (size_t)m_num_nodes * (size_t)m_num_inputs_per_node;

    // ── Weight updates — single flat loop ──
    for (size_t k = 0; k < total_w; ++k) {
      T gradient = m_grad_accum[k] * batch_size_inv;
      gradient = std::clamp(gradient, -gradientClipValue, gradientClipValue);

      m_sq_grad_avg[k] = (rmsPropDecay * m_sq_grad_avg[k]) +
                          (rmsPropDecayInv * gradient * gradient);
      m_sq_grad_avg[k] = std::min(m_sq_grad_avg[k], maxSquaredGradAvg);

      T adj_lr = static_cast<T>(learning_rate) /
                 (std::sqrt(m_sq_grad_avg[k]) + static_cast<T>(rmsPropEpsilon));
      adj_lr = std::min(adj_lr, maxAdjustedLR);

      m_weights[k] -= adj_lr * gradient;
      m_grad_accum[k] = T(0);
    }

    // ── Bias updates ──
    for (size_t i = 0; i < m_num_nodes; ++i) {
      T bg = m_bias_grad_accum[i] * batch_size_inv;
      bg = std::clamp(bg, -gradientClipValue, gradientClipValue);

      m_bias_sq_grad_avg[i] = (rmsPropDecay * m_bias_sq_grad_avg[i]) +
                               (rmsPropDecayInv * bg * bg);
      m_bias_sq_grad_avg[i] = std::min(m_bias_sq_grad_avg[i], maxSquaredGradAvg);

      T adj_lr = static_cast<T>(learning_rate) /
                 (std::sqrt(m_bias_sq_grad_avg[i]) + static_cast<T>(rmsPropEpsilon));
      adj_lr = std::min(adj_lr, maxAdjustedLR);

      m_biases[i] -= adj_lr * bg;
      m_bias_grad_accum[i] = T(0);
    }
  }


  // ════════════════════════════════════════════════════════════════════
  //  Gradient utilities
  // ════════════════════════════════════════════════════════════════════

  float GetGradSumSquared(float batch_size_inv) {
    T sumsq = T(0);
    const size_t total_w = (size_t)m_num_nodes * (size_t)m_num_inputs_per_node;
    for (size_t k = 0; k < total_w; ++k) {
      T scaled = m_grad_accum[k] * batch_size_inv;
      sumsq += scaled * scaled;
    }
    return sumsq;
  }

  void ScaleAccumulatedGradients(T clip_coef) {
    const size_t total_w = (size_t)m_num_nodes * (size_t)m_num_inputs_per_node;
    for (size_t k = 0; k < total_w; ++k) {
      m_grad_accum[k] *= clip_coef;
    }
    for (size_t i = 0; i < m_num_nodes; ++i) {
      m_bias_grad_accum[i] *= clip_coef;
    }
  }


  // ════════════════════════════════════════════════════════════════════
  //  Weight update variants
  // ════════════════════════════════════════════════════════════════════

  /**
   * @brief Immediate per-sample weight update OR gradient accumulation
   */
  void UpdateWeights(const std::vector<T>& input,
                     const std::vector<T>& deriv_error,
                     float learning_rate,
                     std::vector<T>* deltas,
                     bool accumulate = false) {
    if (accumulate) {
      AccumulateGradients(input, deriv_error, deltas);
    } else {
      assert(input.size() == m_num_inputs_per_node);
      assert(deriv_error.size() == m_num_nodes);
      deltas->assign(m_num_inputs_per_node, T(0));

      T *w_ptr = m_weights.data();
      for (size_t i = 0; i < m_num_nodes; ++i) {
        T error_signal = deriv_error[i] *
                         m_deriv_activation_function(m_inner_products[i]);

        for (size_t j = 0; j < m_num_inputs_per_node; ++j) {
          (*deltas)[j] += error_signal * w_ptr[j];
          w_ptr[j] += static_cast<T>(learning_rate * (-(error_signal * input[j])));
        }
        w_ptr += m_num_inputs_per_node;
      }
    }
  }

  /**
   * @brief Calculate gradients without updating weights (for CalcGradients path)
   */
  void CalcGradients(const std::vector<T> &input,
                     const std::vector<T> &deriv_error,
                     std::vector<T> *deltas) {
    assert(input.size() == m_num_inputs_per_node);
    assert(deriv_error.size() == m_num_nodes);
    deltas->assign(m_num_inputs_per_node, T(0));

    const T *w_ptr = m_weights.data();
    for (size_t i = 0; i < m_num_nodes; ++i) {
      T error_signal = deriv_error[i] *
                       m_deriv_activation_function(m_inner_products[i]);
      for (size_t j = 0; j < m_num_inputs_per_node; ++j) {
        (*deltas)[j] += error_signal * w_ptr[j];
      }
      w_ptr += m_num_inputs_per_node;
    }
    grads = *deltas;
  }


  // ════════════════════════════════════════════════════════════════════
  //  Smooth update (target network polyak averaging)
  // ════════════════════════════════════════════════════════════════════

  inline void SmoothUpdateWeights(Layer<T> &source,
                                  const float alpha,
                                  const float alphaInv) {
    assert(m_weights.size() == source.m_weights.size());
    for (size_t k = 0; k < m_weights.size(); ++k) {
      m_weights[k] = (alphaInv * m_weights[k]) + (alpha * source.m_weights[k]);
    }
    for (size_t i = 0; i < m_num_nodes; ++i) {
      m_biases[i] = (alphaInv * m_biases[i]) + (alpha * source.m_biases[i]);
    }
  }


  // ════════════════════════════════════════════════════════════════════
  //  Diagnostics
  // ════════════════════════════════════════════════════════════════════

  T getWeightNorm() const {
    T sum_sq = T(0);
    for (const T& w : m_weights) sum_sq += w * w;
    return std::sqrt(sum_sq);
  }

  void ResetOptimizerState() {
    std::fill(m_sq_grad_avg.begin(), m_sq_grad_avg.end(), T(0));
    std::fill(m_bias_sq_grad_avg.begin(), m_bias_sq_grad_avg.end(), T(0));
  }

  bool CheckAndFixWeights() {
    bool had_corruption = false;
    for (size_t k = 0; k < m_weights.size(); ++k) {
      if (std::isinf(m_weights[k]) || std::isnan(m_weights[k])) {
        m_weights[k] = T(0);
        m_sq_grad_avg[k] = T(0);
        had_corruption = true;
      }
    }
    for (size_t i = 0; i < m_num_nodes; ++i) {
      if (std::isinf(m_biases[i]) || std::isnan(m_biases[i])) {
        m_biases[i] = T(0);
        m_bias_sq_grad_avg[i] = T(0);
        had_corruption = true;
      }
    }
    return had_corruption;
  }


  // ════════════════════════════════════════════════════════════════════
  //  Initialization
  // ════════════════════════════════════════════════════════════════════

  void InitXavier() {
    float limit = 1.0f;
    switch (m_activation_function_type) {
      case ACTIVATION_FUNCTIONS::SIGMOID:
      case ACTIVATION_FUNCTIONS::TANH:
        limit = std::sqrt(6.0f / (m_num_inputs_per_node + m_num_nodes));
        break;
      case ACTIVATION_FUNCTIONS::RELU:
        limit = std::sqrt(6.0f / m_num_inputs_per_node);  // He init
        break;
      default:
        limit = std::sqrt(6.0f / (m_num_inputs_per_node + m_num_nodes));
        break;
    }
    utils::gen_rand<T> randf(limit);
    for (T& w : m_weights) {
      w = randf();
    }
  }


  // ════════════════════════════════════════════════════════════════════
  //  Weight get/set (compatibility with MLP SetLayerWeights)
  // ════════════════════════════════════════════════════════════════════

  /**
   * @brief Set weights from 2D vector [node][input] format
   */
  void SetWeights(std::vector<std::vector<T>>& weights) {
    assert(weights.size() <= m_num_nodes);
    for (size_t i = 0; i < weights.size(); ++i) {
      assert(weights[i].size() == m_num_inputs_per_node);
      T *row = m_weights.data() + i * m_num_inputs_per_node;
      std::copy(weights[i].begin(), weights[i].end(), row);
    }
  }

  /**
   * @brief Get weights as 2D vector [node][input] format
   */
  std::vector<std::vector<T>> GetWeights2D() const {
    std::vector<std::vector<T>> result(m_num_nodes);
    for (size_t i = 0; i < m_num_nodes; ++i) {
      const T *row = m_weights.data() + i * m_num_inputs_per_node;
      result[i].assign(row, row + m_num_inputs_per_node);
    }
    return result;
  }

  void SetGrads(const std::vector<T>& newGrads) { grads = newGrads; }
  std::vector<T>& GetGrads() { return grads; }


  // ════════════════════════════════════════════════════════════════════
  //  Save / Load — backward-compatible with old per-node binary format
  // ════════════════════════════════════════════════════════════════════
  //
  //  Old format per node:  [m_num_inputs(size_t)] [bias(T)] [weights(T * num_inputs)]
  //  We read/write in that exact order, just from flat arrays.

#if ENABLE_SAVE_SD

  bool SaveLayerSD(File &file) const {
    if (file.write((char*)&m_num_nodes, sizeof(m_num_nodes)) != sizeof(m_num_nodes)) return false;
    if (file.write((char*)&m_num_inputs_per_node, sizeof(m_num_inputs_per_node)) != sizeof(m_num_inputs_per_node)) return false;
    if (file.write((char*)&m_activation_function_type, sizeof(m_activation_function_type)) != sizeof(m_activation_function_type)) return false;

    size_t n_inputs = m_num_inputs_per_node;
    for (size_t i = 0; i < m_num_nodes; ++i) {
      if (file.write((char*)&n_inputs, sizeof(n_inputs)) != sizeof(n_inputs)) return false;
      if (file.write((char*)&m_biases[i], sizeof(T)) != sizeof(T)) return false;
      const T *row = m_weights.data() + i * m_num_inputs_per_node;
      int dataSize = m_num_inputs_per_node * sizeof(T);
      if (file.write((char*)row, dataSize) != dataSize) return false;
    }
    return true;
  }

  bool LoadLayerSD(File &file) {
    if (file.read((uint8_t*)&m_num_nodes, sizeof(m_num_nodes)) != sizeof(m_num_nodes)) return false;
    if (file.read((uint8_t*)&m_num_inputs_per_node, sizeof(m_num_inputs_per_node)) != sizeof(m_num_inputs_per_node)) return false;
    if (file.read((uint8_t*)&m_activation_function_type, sizeof(m_activation_function_type)) != sizeof(m_activation_function_type)) return false;

    std::pair<activation_func_t<T>, activation_func_t<T>> *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
        GetActivationFunctionPair(m_activation_function_type, &pair);
    if (!ret_val) return false;
    m_activation_function = pair->first;
    m_deriv_activation_function = pair->second;

    const size_t total_w = (size_t)m_num_nodes * (size_t)m_num_inputs_per_node;
    m_weights.resize(total_w);
    m_biases.resize(m_num_nodes);
    m_inner_products.resize(m_num_nodes);
    m_grad_accum.resize(total_w, T(0));
    m_bias_grad_accum.resize(m_num_nodes, T(0));
    m_sq_grad_avg.resize(total_w, T(0));
    m_bias_sq_grad_avg.resize(m_num_nodes, T(0));

    for (size_t i = 0; i < m_num_nodes; ++i) {
      size_t n_inputs;
      if (file.read((uint8_t*)&n_inputs, sizeof(n_inputs)) != sizeof(n_inputs)) return false;
      if (file.read((uint8_t*)&m_biases[i], sizeof(T)) != sizeof(T)) return false;
      T *row = m_weights.data() + i * m_num_inputs_per_node;
      int dataSize = m_num_inputs_per_node * sizeof(T);
      if (file.read((uint8_t*)row, dataSize) != dataSize) return false;
    }
    return true;
  }

#endif

#ifdef ENABLE_SAVE

  bool SaveLayer(FILE *file) const {
    if (fwrite(&m_num_nodes, sizeof(m_num_nodes), 1, file) != 1) return false;
    if (fwrite(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file) != 1) return false;
    if (fwrite(&m_activation_function_type, sizeof(ACTIVATION_FUNCTIONS), 1, file) != 1) return false;

    size_t n_inputs = m_num_inputs_per_node;
    for (size_t i = 0; i < m_num_nodes; ++i) {
      if (fwrite(&n_inputs, sizeof(n_inputs), 1, file) != 1) return false;
      if (fwrite(&m_biases[i], sizeof(T), 1, file) != 1) return false;
      const T *row = m_weights.data() + i * m_num_inputs_per_node;
      if (fwrite(row, sizeof(T), m_num_inputs_per_node, file) != m_num_inputs_per_node) return false;
    }
    return true;
  }

  bool LoadLayer(FILE *file) {
    if (fread(&m_num_nodes, sizeof(m_num_nodes), 1, file) != 1) return false;
    if (fread(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file) != 1) return false;
    if (fread(&m_activation_function_type, sizeof(ACTIVATION_FUNCTIONS), 1, file) != 1) return false;

    std::pair<activation_func_t<T>, activation_func_t<T>> *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
        GetActivationFunctionPair(m_activation_function_type, &pair);
    if (!ret_val) return false;
    m_activation_function = pair->first;
    m_deriv_activation_function = pair->second;

    const size_t total_w = (size_t)m_num_nodes * (size_t)m_num_inputs_per_node;
    m_weights.resize(total_w);
    m_biases.resize(m_num_nodes);
    m_inner_products.resize(m_num_nodes);
    m_grad_accum.resize(total_w, T(0));
    m_bias_grad_accum.resize(m_num_nodes, T(0));
    m_sq_grad_avg.resize(total_w, T(0));
    m_bias_sq_grad_avg.resize(m_num_nodes, T(0));

    for (size_t i = 0; i < m_num_nodes; ++i) {
      size_t n_inputs;
      if (fread(&n_inputs, sizeof(n_inputs), 1, file) != 1) return false;
      if (fread(&m_biases[i], sizeof(T), 1, file) != 1) return false;
      T *row = m_weights.data() + i * m_num_inputs_per_node;
      if (fread(row, sizeof(T), m_num_inputs_per_node, file) != m_num_inputs_per_node) return false;
    }
    return true;
  }

#endif


  // ════════════════════════════════════════════════════════════════════
  //  Data members — all public for direct MLP access
  // ════════════════════════════════════════════════════════════════════

  // Core storage — flat, contiguous
  std::vector<T> m_weights;           ///< [num_nodes * num_inputs], row-major
  std::vector<T> m_biases;            ///< [num_nodes]
  std::vector<T> m_inner_products;    ///< [num_nodes] cached pre-activation

  // Optimizer state
  std::vector<T> m_grad_accum;        ///< [num_nodes * num_inputs]
  std::vector<T> m_bias_grad_accum;   ///< [num_nodes]
  std::vector<T> m_sq_grad_avg;       ///< [num_nodes * num_inputs] RMSProp
  std::vector<T> m_bias_sq_grad_avg;  ///< [num_nodes]

  std::vector<T> cachedOutputs;

  size_t m_num_inputs_per_node{ 0 };
  size_t m_num_nodes{ 0 };

protected:
  ACTIVATION_FUNCTIONS m_activation_function_type;
  activation_func_t<T> m_activation_function{ nullptr };
  activation_func_t<T> m_deriv_activation_function{ nullptr };

  bool m_cacheOutputs{ false };
  std::vector<T> grads;
};

#endif // LAYER_H