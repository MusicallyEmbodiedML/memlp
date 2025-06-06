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
#include <unordered_map>
// #include <string>


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
    LOSS_MSE /**< Mean Squared Error loss function */
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
    T one_over_n_elem = 1. / n_elem;
    
    for (unsigned int j = 0; j < actual.size(); j++) {
        //TODO CK separate out diff for efficiency, replace pow with diff*diff
          accum_loss += std::pow((expected[j] - actual[j]), 2)
                * one_over_n_elem;
          loss_deriv[j] =
              -2 * one_over_n_elem
              * (expected[j] - actual[j]) * sampleSizeReciprocal;
    }
    accum_loss *= sampleSizeReciprocal;

    return accum_loss;
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
    };

    std::unordered_map<
        LOSS_FUNCTIONS,
        loss_func_t<T>
    > loss_functions_map; /**< Map storing loss functions */
};

}  // namespace loss

#endif  // __LOSS_H__
