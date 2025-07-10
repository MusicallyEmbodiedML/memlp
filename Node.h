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

#endif

#include "Utils.h"

#include <vector>
#include <cassert> // for assert()
#include <numeric>
#include <algorithm>

#include "pico.h"

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
        m_num_inputs = 0;
        m_bias = 0.0;
        m_weights.clear();
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
    }

    /**
     * @brief Randomizes weights with Gaussian noise
     * @param variance The variance of the Gaussian distribution
     */
    void WeightRandomisation(float variance) {
        std::transform(m_weights.begin(),
                       m_weights.end(),
                       m_weights.begin(),
                       utils::gen_randn<T>(variance));
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
    void SetWeights( std::vector<T> & weights ){
        // check size of the weights vector
        assert(weights.size() == m_num_inputs);
        m_weights = weights;
    }

    /**
     * @brief Updates weights using exponential moving average
     * @param incomingWeights New weights to blend with current weights
     * @param alpha Learning rate for new weights
     * @param alphaInv Learning rate for current weights (typically 1-alpha)
     */
    void __force_inline SmoothUpdateWeights(std::vector<T> & incomingWeights, const float alpha, const float alphaInv) {
        assert(incomingWeights.size() == m_weights.size());
        for(size_t i = 0; i < m_weights.size(); i++) {
            m_weights[i] = (alphaInv * m_weights[i]) + (alpha * incomingWeights[i]);
            // #sleep_us(50);
        }

    }

    /**
     * @brief Gets the size of the weights vector
     * @return Number of weights
     */
    size_t __force_inline GetWeightsVectorSize() const {
        return m_weights.size();
    }

    /**
     * @brief Computes inner product of input with weights
     * @param input Vector of input values
     * @return Inner product result
     */
    T __force_inline GetInputInnerProdWithWeights(const std::vector<T> &input) {

        static const T kInit(0);

        // assert(input.size() == m_weights.size());

        // T res = std::inner_product(begin(input),
        //                            end(input),
        //                            begin(m_weights),
        //                            kInit);
        // // *output = res;
        // inner_prod = res;

        T res=0;
        for(size_t j=0; j < input.size(); j++) {
          res += input[j] * m_weights[j];
        }

        // *output = res;
        inner_prod = res;

        return inner_prod;
    }

    /**
     * @brief Computes node output using specified activation function
     * @param input Input vector
     * @param activation_function Activation function to use
     * @param output Pointer to store the output value
     */
    void __force_inline GetOutputAfterActivationFunction(const std::vector<T> &input,
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
    void  GetBooleanOutput(const std::vector<T> &input,
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
    void __force_inline UpdateWeights(const std::vector<T> &x,
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
    void __force_inline UpdateWeight(int weight_id,
                      float increment,
                      float learning_rate) {
        m_weights[weight_id] += static_cast<T>(learning_rate*increment);
    }

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
    T inner_prod;            /**< Cached inner product value */

private:
    Node<T>& operator=(Node<T> const &) = delete; /**< Deleted assignment operator */
};

#endif //NODE_H
