/**
 * @file Node.h
 * @brief Neural network node implementation with weight management and activation functions
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This code is derived from David Alberto Nogueira's MLP project:
 * https://github.com/davidalbertonogueira/MLP
 */

#ifndef NODE_H
#define NODE_H

#ifdef ARDUINO

#include <Arduino.h>
#include <VFS.h>
#include <LittleFS.h>
#define ENABLE_SAVE    1
#include "pico.h"

#endif

#include "Utils.h"

#include <vector>
#include <cassert> // for assert()
#include <numeric>
#include <algorithm>
#include <span>

#define CONSTANT_WEIGHT_INITIALIZATION 0

/**
 * @brief Definition of activation function pointer
 * @tparam T The numeric type used for calculations
 */
template<typename T>
using activation_func_t = T(*)(T);

/**
 * @class Node
 * @brief Represents a single neural network node with weights and activation capabilities
 * @tparam T The numeric type used for calculations (typically float or double)
 */
template <typename T>
class Node {
public:
    /**
     * @brief Default constructor
     */
    Node() {
        m_num_inputs = 0;
        m_bias = 0;
        m_weights.clear();
        squared_gradient_avg.clear();
    };

    /**
     * @brief Constructor with initialization parameters
     * @param num_inputs Number of input connections to the node
     * @param use_constant_weight_init Flag to use constant weight initialization
     * @param constant_weight_init Value for constant weight initialization
     */
    Node(int num_inputs,
         bool use_constant_weight_init = true,
         T constant_weight_init = 0.5) {
        m_num_inputs = num_inputs;
        m_bias = 0.0;
        m_weights.clear();
        //initialize weight vector
        WeightInitialization(m_num_inputs,
                             use_constant_weight_init,
                             constant_weight_init);
    };

    ~Node() {
    };

    /**
     * @brief Initializes the node's weights
     * @param num_inputs Number of input connections
     * @param use_constant_weight_init Flag to use constant weight initialization
     * @param constant_weight_init Value for constant weight initialization
     */
    void WeightInitialization(int num_inputs,
                              bool use_constant_weight_init = true,
                              T constant_weight_init = 0.5) {
        m_num_inputs = num_inputs;
        //initialize weight vector
        if (use_constant_weight_init) {
            m_weights.resize(m_num_inputs, constant_weight_init);
        } else {
            m_weights.resize(m_num_inputs);
            std::generate_n(m_weights.begin(),
                            m_num_inputs,
                            utils::gen_rand<T>());
        }
        squared_gradient_avg.resize(m_num_inputs);
        std::fill(squared_gradient_avg.begin(), squared_gradient_avg.end(), 0.f);
    }

    /**
     * @brief Randomizes weights with Gaussian noise
     * @param variance The variance of the Gaussian distribution
     */
    void WeightRandomisation(const float variance) {
        std::transform(m_weights.begin(),
                       m_weights.end(),
                       m_weights.begin(),
                       utils::gen_randn<T>(variance));
    }

    /**
     * @brief Initialize gradient accumulator
     */
    void InitializeGradientAccumulator() {
        m_gradient_accumulator.clear();
        m_gradient_accumulator.resize(m_weights.size(), 0);
        m_bias_gradient_accumulator = 0;
    }
    
    /**
     * @brief Accumulate gradients without updating weights
     * @param x Input vector
     * @param error Error signal from backpropagation
     * @param learning_rate Not used here, kept for compatibility
     */
    inline void AccumulateGradients(std::span<const T> x,
                                   T error) {
        assert(x.size() == m_weights.size());
        for (size_t i = 0; i < m_weights.size(); i++) {
            m_gradient_accumulator[i] += x[i] * error;
        }
        m_bias_gradient_accumulator += error;         
    }

    //     /**
    //  * @brief Apply accumulated gradients and clear accumulator
    //  * @param learning_rate Learning rate for weight update
    //  * @param batch_size Size of the batch for averaging
    //  */
    // inline void ApplyAccumulatedGradients(float learning_rate, T batch_size_inv) {
    //     T scale = learning_rate * batch_size_inv;
    //     for (size_t i = 0; i < m_weights.size(); i++) {
    //         m_weights[i] -= m_gradient_accumulator[i] * scale;
    //         m_gradient_accumulator[i] = 0;  // Reset accumulator
    //     }
    // }


    float rmsPropDecay = 0.9f;
    float rmsPropDecayInv = 0.1f;
    float rmsPropEpsilon = 1e-7f;
    inline void ApplyAccumulatedGradients(float learning_rate, T batch_size_inv) {
        for (size_t i = 0; i < m_weights.size(); i++) {
            T gradient = m_gradient_accumulator[i] * batch_size_inv;
            squared_gradient_avg[i] = (rmsPropDecay * squared_gradient_avg[i]) + 
                                         (rmsPropDecayInv * gradient * gradient);
            float adjusted_learning_rate = learning_rate / 
                                   (std::sqrt(squared_gradient_avg[i]) + rmsPropEpsilon);  
            m_weights[i] -= adjusted_learning_rate * gradient;
            
            m_gradient_accumulator[i] = 0.f;  // Reset accumulator
        }
        T bias_gradient = m_bias_gradient_accumulator * batch_size_inv;
        bias_squared_gradient_avg = (rmsPropDecay * bias_squared_gradient_avg) + 
                                    (rmsPropDecayInv * bias_gradient * bias_gradient);
        float bias_adjusted_lr = learning_rate / (std::sqrt(bias_squared_gradient_avg) + rmsPropEpsilon);
        m_bias -= bias_adjusted_lr * bias_gradient;
        m_bias_gradient_accumulator = 0;
        // printf("Bias: %f\n", m_bias);
    }


    inline float GetGradSumSquared(T batch_size_inv) {
        T sumsq = 0;
        for (size_t i = 0; i < m_gradient_accumulator.size(); i++) {
            T scaledGrad = m_gradient_accumulator[i] * batch_size_inv;
            sumsq +=  scaledGrad*scaledGrad;
        }
        return sumsq;
    }

    void ScaleAccumulatedGradients(T clip_coef) {
        for (size_t i = 0; i < m_gradient_accumulator.size(); i++) {
            m_gradient_accumulator[i] *= clip_coef;
        }
    }

   /**
     * @brief Clear gradient accumulator
     */
    inline void ClearGradientAccumulator() {
        std::fill(m_gradient_accumulator.begin(), m_gradient_accumulator.end(), 0);
    }    

    /**
     * @brief Gets the number of inputs to this node
     * @return Number of inputs
     */
    int GetInputSize() const {
        return m_num_inputs;
    }

    /**
     * @brief Sets the number of inputs to this node
     * @param num_inputs New number of inputs
     */
    void SetInputSize(int num_inputs) {
        m_num_inputs = num_inputs;
    }

    /**
     * @brief Gets the node's bias value
     * @return Current bias value
     */
    T GetBias() const {
        return m_bias;
    }

    /**
     * @brief Sets the node's bias value
     * @param bias New bias value
     */
    void SetBias(T bias) {
        m_bias = bias;
    }

    /**
     * @brief Gets reference to the weight vector
     * @return Reference to weights vector
     */
    std::vector<T> & GetWeights() {
        return m_weights;
    }

    /**
     * @brief Gets const reference to the weight vector
     * @return Const reference to weights vector
     */
    const std::vector<T> & GetWeights() const {
        return m_weights;
    }

    /**
     * @brief Sets new weights for the node
     * @param weights Vector of new weights
     */
    void SetWeights( std::span<T> weights ){
        // check size of the weights vector
        assert(weights.size() == m_num_inputs);
        // m_weights = weights;
        m_weights.assign(weights.begin(), weights.end());
    }

    /**
     * @brief Updates weights using exponential moving average
     * @param incomingWeights New weights to blend with current weights
     * @param alpha Learning rate for new weights
     * @param alphaInv Learning rate for current weights (typically 1-alpha)
     */
    inline void SmoothUpdateWeights(std::span<T> incomingWeights, const float alpha, const float alphaInv) {
        assert(incomingWeights.size() == m_weights.size());
        for(size_t i = 0; i < m_weights.size(); i++) {
            m_weights[i] = (alphaInv * m_weights[i]) + (alpha * incomingWeights[i]);
        }

    }

    /**
     * @brief Gets the size of the weights vector
     * @return Number of weights
     */
    inline size_t GetWeightsVectorSize() const {
        return m_weights.size();
    }

    /**
     * @brief Computes inner product of input with weights
     * @param input Vector of input values
     * @return Inner product result
     */
    inline T GetInputInnerProdWithWeights(std::span<const T> &input) {


        T res=0;
        for(size_t j=0; j < input.size(); j++) {
          res += input[j] * m_weights[j];
        }
        res += m_bias;
        inner_prod = res;

        return inner_prod;
    }

    /**
     * @brief Computes node output using specified activation function
     * @param input Input vector
     * @param activation_function Activation function to use
     * @param output Pointer to store the output value
     */
    inline void GetOutputAfterActivationFunction(std::span<const T> input,
                                          MLP_ACTIVATION_FN activation_func_t<T> activation_function,
                                          T * output) {
        // T inner_prod = 0.0;
        GetInputInnerProdWithWeights(input);
        *output = activation_function(inner_prod);
    }

    /**
     * @brief Computes binary output based on activation threshold
     * @param input Input vector
     * @param activation_function Activation function to use
     * @param bool_output Pointer to store the binary output
     * @param threshold Threshold value for binary decision
     */
    void  GetBooleanOutput(std::vector<const T> input,
                          MLP_ACTIVATION_FN activation_func_t<T> activation_function,
                          bool * bool_output,
                          T threshold = 0.5) {
        T value;
        GetOutputAfterActivationFunction(input, activation_function, &value);
        *bool_output = (value > threshold) ? true : false;
    };

    /**
     * @brief Updates weights based on error and learning rate
     * @param x Input vector
     * @param error Error value
     * @param learning_rate Learning rate for weight update
     */
    inline void UpdateWeights(std::span<const T> x,
                       T error,
                       T learning_rate) {
        assert(x.size() == m_weights.size());
        for (size_t i = 0; i < m_weights.size(); i++)
            m_weights[i] += x[i] * learning_rate *  error;
    };

    /**
     * @brief Updates a single weight
     * @param weight_id Index of weight to update
     * @param increment Amount to increment the weight
     * @param learning_rate Learning rate for weight update
     */
    inline void UpdateWeight(int weight_id,
                      float increment,
                      float learning_rate) {
        m_weights[weight_id] += static_cast<T>(learning_rate*increment);
    }
#if ENABLE_SAVE_SD

bool SaveNodeSD(File &file) const {
    if (file.write((char*)&m_num_inputs, sizeof(m_num_inputs)) != sizeof(m_num_inputs)) {
      return false;
    }
    if (file.write((char*)&m_num_inputs, sizeof(m_num_inputs)) != sizeof(m_num_inputs)) {
      return false;
    }

    if (!m_weights.empty()) {
        int dataSize = m_weights.size() * sizeof(m_weights[0]);
        if (file.write((char*)&m_weights[0], dataSize) != dataSize) {
            return false;
        }
    }
    return true;
};

bool LoadNodeSD(File &file) {
    m_weights.clear();

    if (file.read((uint8_t*)&m_num_inputs, sizeof(m_num_inputs)) != sizeof(m_num_inputs)) {
        return false;
    }
    if (file.read((uint8_t*)&m_bias, sizeof(m_bias)) != sizeof(m_bias)) {
        return false;
    }

    m_weights.resize(m_num_inputs);
    if (!m_weights.empty()) {
        int dataSize = m_weights.size() * sizeof(T);
        if (file.read((uint8_t*)&m_weights[0], dataSize) != dataSize) {
            return false;
        }
    }
    return true;
};


#endif

#if ENABLE_SAVE
    /**
     * @brief Saves node state to file
     * @param file File pointer for saving
     * @return true if save was successful, false if there was an error
     */
    bool SaveNode(FILE * file) const {
        if (fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file) != 1) {
            return false;
        }
        if (fwrite(&m_bias, sizeof(m_bias), 1, file) != 1) {
            return false;
        }
        if (!m_weights.empty()) {
            if (fwrite(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file) != m_weights.size()) {
                return false;
            }
        }
        return true;
    };

    /**
     * @brief Loads node state from file
     * @param file File pointer for loading
     * @return true if load was successful, false if there was an error
     */
    bool LoadNode(FILE * file) {
        m_weights.clear();

        if (fread(&m_num_inputs, sizeof(m_num_inputs), 1, file) != 1) {
            return false;
        }
        if (fread(&m_bias, sizeof(m_bias), 1, file) != 1) {
            return false;
        }

        m_weights.resize(m_num_inputs);
        if (!m_weights.empty()) {
            if (fread(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file) != m_weights.size()) {
                return false;
            }
        }
        return true;
    };
#endif

    size_t m_num_inputs{ 0 }; /**< Number of inputs to this node */
    T m_bias{ 0.0 };         /**< Bias value for this node */
    std::vector<T> m_weights; /**< Vector of input weights */

    /**
     * @brief Accumulated gradients for batch training
     */
    std::vector<T> m_gradient_accumulator;
    std::vector<T> squared_gradient_avg;    
    T m_bias_gradient_accumulator{0}; 
    T bias_squared_gradient_avg=0;

    inline T GetInnerProd() const {
        return inner_prod;
    }
private:
    Node<T>& operator=(Node<T> const &) = delete; /**< Deleted assignment operator */
    T inner_prod;            /**< Cached inner product value */
};

#endif //NODE_H
