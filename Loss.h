/**
 * @file Loss.h
 * @brief Loss functions and management for machine learning operations
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This code is derived from David Alberto Nogueira's MLP project:
 * https://github.com/davidalbertonogueira/MLP
 */

#ifndef __LOSS_H__
#define __LOSS_H__

#include <vector>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
// #include <string>

// Placement.h defines SMLP_CODE_ATTR as blank if not already bound. A host
// project that wants real placement must #include its own binding header
// (e.g. MemoryDefs.hpp) before any mlp/*.h header, this one included.
#include "Placement.h"
#include "FastMath.h" // smlp::exp_f32/log_f32: tagged, no core-0-bank libm dependency

#if defined(__XS3A__)

#define MLP_LOSS_FN __attribute__(( fptrgroup("mlp_loss") ))

#else

//#pragma message ( "PC compiler definitions enabled - check this is OK" )
#define MLP_LOSS_FN

#endif


namespace loss {

/**
 * @enum LOSS_FUNCTIONS
 * @brief Enumeration of supported loss functions.
 */
enum LOSS_FUNCTIONS {
    LOSS_MSE, /**< Mean Squared Error loss function */
    LOSS_CATEGORICAL_CROSSENTROPY /**< Categorical Cross-Entropy loss function */
};

/**
 * @brief Computes the Mean Squared Error loss between expected and actual values
 * @tparam T The type of the values
 * @param expected Vector of expected values
 * @param actual Vector of actual values
 * @param loss_deriv Vector to store the loss derivatives
 * @param sampleSizeReciprocal Reciprocal of the sample size for normalization
 * @return The computed MSE loss value
 */
template<typename T>
MLP_LOSS_FN
inline T MSE(const std::vector<T> &expected, const std::vector<T> &actual,
             std::vector<T> &loss_deriv, T sampleSizeReciprocal) {

    T accum_loss = 0.;
    T n_elem = actual.size();
    T one_over_n_elem = (T)1. / n_elem;

    for (unsigned int j = 0; j < actual.size(); j++) {
        //TODO CK separate out diff for efficiency, replace pow with diff*diff
        const T diff = expected[j] - actual[j];
          accum_loss += (diff * diff) //std::pow((expected[j] - actual[j]), 2)
                * one_over_n_elem;
          loss_deriv[j] =
              (T)-2 * one_over_n_elem
              * diff * sampleSizeReciprocal;
    }
    accum_loss *= sampleSizeReciprocal;

    return accum_loss;
}

/**
 * @brief Computes stable categorical cross-entropy loss and its logits gradient
 *        from raw pointers, allocating nothing.
 * @tparam T The type of the values
 * @param target Pointer to `n` target class probabilities (one-hot or general)
 * @param logits Pointer to `n` raw (pre-softmax) logits
 * @param logits_grad Pointer to `n` elements receiving softmax(logits) - target,
 *        scaled by `sampleSizeReciprocal`
 * @param n Number of classes
 * @param sampleSizeReciprocal Reciprocal of the sample size for normalization
 * @return The computed categorical cross-entropy loss value
 *
 * Uses the log-sum-exp trick for numerical stability and evaluates the full
 * `-Σ target[i] * log_softmax[i]` contract, so `target` need not be one-hot.
 */
namespace detail {
// float uses the private, hot-path-placed exp/log (see mlp/FastMath.h); other
// types (fixed-point) keep going through std::exp/std::log's own overloads.
template<typename T> SMLP_CODE_ATTR inline T exp(T x) {
    if constexpr (std::is_same_v<T, float>) return ::smlp::exp_f32(x);
    else                                    return std::exp(x);
}
template<typename T> SMLP_CODE_ATTR inline T log(T x) {
    if constexpr (std::is_same_v<T, float>) return ::smlp::log_f32(x);
    else                                    return std::log(x);
}
} // namespace detail

template<typename T>
MLP_LOSS_FN
SMLP_CODE_ATTR
inline T CategoricalCrossEntropyLogits(const T *target, const T *logits, T *logits_grad,
                                        std::size_t n, T sampleSizeReciprocal) {

    // Max-shift the logits before exponentiating (log-sum-exp trick).
    T max_logit = logits[0];
    for (std::size_t i = 1; i < n; i++) {
        if constexpr (std::is_same_v<T, float>) {
            max_logit = __builtin_fmaxf(max_logit, logits[i]); // single vmaxnm, no flag round-trip
        } else if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        // Hot path (float). Domain-restricted exp/log (mlp/FastMath.h): after the
        // max shift every exponent is <= 0 and the sum lies in [1, n]. The shifted
        // exponentials are parked in logits_grad (it is exactly `n` floats of
        // caller scratch) so the softmax is e_i / sum, not a second round of exp().
        float sum_exp = 0.0f;
        // Also accumulates sum(t) and sum(t*l) here so the cross-entropy needs no
        // loop of its own: -sum t*(l - lse) = lse*sum(t) - sum(t*l), valid for any
        // (not just one-hot) target.
        float sum_t = 0.0f, sum_tl = 0.0f;
        // Two elements per iteration, written out by hand (`#pragma GCC unroll`
        // was not honoured here): one exp is a long dependent FMA chain on which
        // the in-order FPU stalls, two independent chains let the scheduler
        // interleave them.
        std::size_t i = 0;
        for (; i + 1 < n; i += 2) {
            const float l0 = logits[i], l1 = logits[i + 1];
            const float e0 = ::smlp::exp_nonpos_f32(l0 - max_logit);
            const float e1 = ::smlp::exp_nonpos_f32(l1 - max_logit);
            logits_grad[i]     = e0;
            logits_grad[i + 1] = e1;
            sum_exp += e0 + e1;
            sum_t   += target[i] + target[i + 1];
            sum_tl   = __builtin_fmaf(target[i + 1], l1, __builtin_fmaf(target[i], l0, sum_tl));
        }
        if (i < n) {
            const float e = ::smlp::exp_nonpos_f32(logits[i] - max_logit);
            logits_grad[i] = e;
            sum_exp += e;
            sum_t   += target[i];
            sum_tl   = __builtin_fmaf(target[i], logits[i], sum_tl);
        }
        const float log_sum_exp = max_logit + ::smlp::log_ge1_f32(sum_exp);
        const float loss = log_sum_exp * sum_t - sum_tl;
        const float inv_sum = 1.0f / sum_exp;
        for (std::size_t i = 0; i < n; i++) {
            logits_grad[i] = (logits_grad[i] * inv_sum - target[i]) * sampleSizeReciprocal;
        }
        return loss * sampleSizeReciprocal;
    } else {
        // log-sum-exp(logits) computed from the shifted exponentials.
        T sum_exp = T(0);
        for (std::size_t i = 0; i < n; i++) {
            sum_exp += detail::exp(logits[i] - max_logit);
        }
        T log_sum_exp = max_logit + detail::log(sum_exp);

        // Full cross-entropy sum, not just the one-hot fast path: supports any
        // target distribution (probabilities), not only one-hot class labels.
        T loss = T(0);
        for (std::size_t i = 0; i < n; i++) {
            loss -= target[i] * (logits[i] - log_sum_exp);
        }

        // Gradient w.r.t. logits is softmax(logits) - target, scaled per-sample.
        for (std::size_t i = 0; i < n; i++) {
            T softmax_i = detail::exp(logits[i] - log_sum_exp);
            logits_grad[i] = (softmax_i - target[i]) * sampleSizeReciprocal;
        }

        return loss * sampleSizeReciprocal;
    }
}

/**
 * @brief Computes the Categorical Cross-Entropy loss between expected and actual values
 * @tparam T The type of the values
 * @param expected Vector of expected class-probability values (one-hot or general)
 * @param actual Vector of raw logits (pre-softmax)
 * @param loss_deriv Vector to store the loss derivatives
 * @param sampleSizeReciprocal Reciprocal of the sample size for normalization
 * @return The computed categorical cross-entropy loss value
 *
 * Thin delegator over the allocation-free ::CategoricalCrossEntropyLogits().
 */
template<typename T>
MLP_LOSS_FN
inline T CategoricalCrossEntropy(const std::vector<T> &expected, const std::vector<T> &actual,
                                std::vector<T> &loss_deriv, T sampleSizeReciprocal) {
    return CategoricalCrossEntropyLogits(expected.data(), actual.data(), loss_deriv.data(),
                                          actual.size(), sampleSizeReciprocal);
}

/**
 * @typedef loss_func_t
 * @brief Type definition for loss function pointers
 * @tparam T The type of the values
 */
template<typename T>
using loss_func_t = T(*)(const std::vector<T> &, const std::vector<T> &, std::vector<T> &, T);

/**
 * @class LossFunctionsManager
 * @brief Manages loss functions and their access
 * @tparam T The type of the values used in loss calculations
 */
template<typename T>
class LossFunctionsManager {
 public:
    /**
     * @brief Retrieves a loss function by its identifier
     * @param loss_name The identifier of the loss function
     * @param loss_fun Pointer to store the retrieved loss function
     * @return True if the loss function is found, false otherwise
     */
    bool GetLossFunction(const LOSS_FUNCTIONS loss_name,
                         loss_func_t<T> *loss_fun) {

        auto iter = loss_functions_map.find(loss_name);
        if (iter != loss_functions_map.end()) {
            *loss_fun = iter->second;
        } else {
            return false;
        }
        return true;
    }

    /**
     * @brief Retrieves the singleton instance of LossFunctionsManager
     * @return The singleton instance
     */
    static LossFunctionsManager & Singleton() {
        static LossFunctionsManager instance;
        return instance;
    }

 private:
    /**
     * @brief Adds a new loss function to the manager
     * @param function_name The identifier for the loss function
     * @param function The loss function to add
     */
    void AddNew(LOSS_FUNCTIONS function_name,
                loss_func_t<T> function) {
        loss_functions_map.insert(
            std::make_pair(function_name, function)
        );
    };

    /**
     * @brief Private constructor for singleton pattern
     */
    LossFunctionsManager() {
        AddNew(LOSS_FUNCTIONS::LOSS_MSE, &MSE<T>);
        AddNew(LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY, &CategoricalCrossEntropy<T>);
    };

    std::unordered_map<
        LOSS_FUNCTIONS,
        loss_func_t<T>
    > loss_functions_map; /**< Map storing loss functions */
};

}  // namespace loss

#endif  // __LOSS_H__
