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
#include <unordered_map>
// #include <string>

// Placement.h defines SMLP_CODE_ATTR as blank if not already bound. A host
// project that wants real placement must #include its own binding header
// (e.g. MemoryDefs.hpp) before any mlp/*.h header, this one included.
#include "Placement.h"

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
template<typename T>
MLP_LOSS_FN
SMLP_CODE_ATTR
inline T CategoricalCrossEntropyLogits(const T *target, const T *logits, T *logits_grad,
                                        std::size_t n, T sampleSizeReciprocal) {

    // Max-shift the logits before exponentiating (log-sum-exp trick).
    T max_logit = logits[0];
    for (std::size_t i = 1; i < n; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    // log-sum-exp(logits) computed from the shifted exponentials.
    T sum_exp = T(0);
    for (std::size_t i = 0; i < n; i++) {
        sum_exp += std::exp(logits[i] - max_logit);
    }
    T log_sum_exp = max_logit + std::log(sum_exp);

    // Full cross-entropy sum, not just the one-hot fast path: supports any
    // target distribution (probabilities), not only one-hot class labels.
    T loss = T(0);
    for (std::size_t i = 0; i < n; i++) {
        loss -= target[i] * (logits[i] - log_sum_exp);
    }

    // Gradient w.r.t. logits is softmax(logits) - target, scaled per-sample.
    for (std::size_t i = 0; i < n; i++) {
        T softmax_i = std::exp(logits[i] - log_sum_exp);
        logits_grad[i] = (softmax_i - target[i]) * sampleSizeReciprocal;
    }

    return loss * sampleSizeReciprocal;
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
