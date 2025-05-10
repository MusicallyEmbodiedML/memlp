/**
 * @file MLP.h
 * @brief Multi-layer perceptron neural network implementation
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This code is derived from David Alberto Nogueira's MLP project:
 * https://github.com/davidalbertonogueira/MLP
 */

#ifndef MLP_H
#define MLP_H

#include "Layer.h"
#include "Utils.h"
#include "Loss.h"
#include "Sample.h"

#include <cstdint>
#include <memory>
#include <string>

#include "Arduino.h"

/**
 * @class MLP
 * @brief Multi-layer perceptron neural network with flexible architecture
 * 
 * This class implements a fully-connected multi-layer perceptron with configurable layers,
 * nodes per layer, and activation functions. It supports both training and inference modes,
 * and includes special features for reinforcement learning applications.
 * 
 * @tparam T The numeric type used for weights and calculations (typically float)
 */
template<typename T>
class MLP {
public:
    /**
     * @brief Data type for training data pairs (features, labels)
     */
    using training_pair_t = std::pair<
      std::vector< std::vector<T> >,
      std::vector< std::vector<T> >
    >;

    /**
     * @brief Data type for storing network weights
     */
    using mlp_weights = std::vector< std::vector <std::vector<T> > >;

    /**
     * @brief Constructs an MLP with specified architecture
     * 
     * @param layers_nodes Vector specifying number of nodes in each layer (including input and output)
     * @param layers_activfuncs Vector of activation functions for each layer (except input)
     * @param loss_function Loss function for training (default: MSE)
     * @param use_constant_weight_init Whether to use constant weight initialization (default: false)
     * @param constant_weight_init Value for constant weight initialization if enabled (default: 0.5)
     */
    MLP(const std::vector<size_t> & layers_nodes,
        const std::vector<ACTIVATION_FUNCTIONS> & layers_activfuncs,
        loss::LOSS_FUNCTIONS loss_function = loss::LOSS_FUNCTIONS::LOSS_MSE,
        bool use_constant_weight_init = false,
        T constant_weight_init = 0.5);

#if ENABLE_SAVE
    MLP(const std::string & filename);
#endif
    ~MLP();

#if ENABLE_SAVE
    void SaveMLPNetwork(const std::string & filename) const;
    void LoadMLPNetwork(const std::string & filename);
#endif

    size_t Serialise(size_t w_head, std::vector<uint8_t> &buffer);
    size_t FromSerialised(size_t w_head, const std::vector<uint8_t> &buffer);

    /**
     * @brief Get predicted outputs for given input
     * 
     * @param input Input feature vector
     * @param output Pointer to store output predictions
     * @param all_layers_activations Optional pointer to store activations of all layers
     */
    void GetOutput(const std::vector<T> &input,
                    std::vector<T> * output,
                    std::vector<std::vector<T>> * all_layers_activations = nullptr);

    /**
     * @brief Determines the output class from network outputs
     * 
     * @param output Network output vector
     * @param class_id Pointer to store the predicted class ID
     */
    void GetOutputClass(const std::vector<T> &output, size_t * class_id) const;

    /**
     * @brief Train the network using batch gradient descent
     * 
     * @param training_sample_set_with_bias Training data pairs
     * @param learning_rate Learning rate for gradient descent
     * @param max_iterations Maximum training iterations
     * @param min_error_cost Minimum error threshold for early stopping
     * @param output_log Whether to output training progress
     * @return Final training error
     */
    T Train(const training_pair_t& training_sample_set_with_bias,
                float learning_rate,
                int max_iterations = 5000,
                float min_error_cost = 0.001,
                bool output_log = true);

    /**
     * @brief Train the network using mini-batch gradient descent
     * 
     * @param training_sample_set_with_bias Training data pairs
     * @param learning_rate Learning rate for gradient descent
     * @param max_iterations Maximum training iterations
     * @param miniBatchSize Size of mini-batches
     * @param min_error_cost Minimum error threshold for early stopping
     * @param output_log Whether to output training progress
     * @return Final training error
     */
    T MiniBatchTrain(const training_pair_t& training_sample_set_with_bias,
                float learning_rate,
                int max_iterations = 5000,
                size_t miniBatchSize=8,
                float min_error_cost = 0.001,
                bool output_log = true);

    /**
     * @deprecated Use Train() with training_pair_t instead
     * @brief Legacy training method using TrainingSample objects
     */
    void Train(const std::vector<TrainingSample<T>> &training_sample_set_with_bias,
                        float learning_rate,
                        int max_iterations = 5000,
                        float min_error_cost = 0.001,
                        bool output_log = true);

    /**
     * @brief Get number of layers in the network
     */
    size_t GetNumLayers();

    /**
     * @brief Get weights for a specific layer
     * @param layer_i Layer index
     */
    std::vector<std::vector<T>> GetLayerWeights( size_t layer_i );

    /**
     * @brief Get all network weights
     * @return 3D vector of weights (layer, node, weight)
     */
    mlp_weights GetWeights();

    /**
     * @brief Set weights for a specific layer
     * @param layer_i Layer index
     * @param weights 2D vector of weights for the layer
     */
    void SetLayerWeights( size_t layer_i, std::vector<std::vector<T>> & weights );

    /**
     * @brief Set all network weights
     * @param weights 3D vector of weights (layer, node, weight)
     */
    void SetWeights(mlp_weights &weights);

    /**
     * @brief Randomize network weights
     */
    void DrawWeights();

    /**
     * @brief Add Gaussian noise to network weights
     * @param speed Standard deviation of the noise
     */
    void MoveWeights(T speed);

    /**
     * @brief Enable/disable caching of layer outputs
     * 
     * Required for backpropagation and some RL algorithms
     * 
     * @param on True to enable caching, false to disable
     */
    void SetCachedLayerOutputs(bool on) {
        for(auto &layer : m_layers) {
            layer.SetCachedOutputs(on);
        }
    }

    /**
     * @brief Perform soft update of network weights (for RL)
     * 
     * Updates this network's weights using exponential moving average with another network's weights.
     * Commonly used in RL for target networks.
     * 
     * @param anotherMLP Source network for weight update
     * @param alpha Learning rate (0-1) for the update
     */
    void __force_inline SmoothUpdateWeights(std::shared_ptr<MLP<T>> anotherMLP, const float alpha) {
        //assuming the other MLP has the same structure
        //calc this once here
        const float alphaInv = 1.f-alpha;

        for(size_t i=0; i < m_layers.size(); i++) {
            m_layers[i].SmoothUpdateWeights(anotherMLP->m_layers[i], alpha, alphaInv);
        }
    }

    /**
     * @brief Calculate gradients through the network (autograd)
     * 
     * Similar to TensorFlow's tf.gradients(), computes gradients of the network
     * with respect to the inputs. Useful for policy gradients in RL.
     * 
     * @param feat Input feature vector
     * @param deriv_error_output Initial gradient at the output layer
     */
    void CalcGradients(std::vector<T> &feat, std::vector<T> & deriv_error_output);

    /**
     * @brief Clear accumulated gradients
     */
    void ClearGradients() {
        for(auto &v: m_layers) {
            v.SetGrads({});
        }
    }



    /**
     * @brief Backpropagation of loss through the network
     * 
     * @param feat Input feature vector
     * @param loss Loss values
     * @param learning_rate Learning rate for weight updates
     */
    void ApplyLoss(std::vector<T> feat,
          std::vector<T> loss,
          float learning_rate);


    /**
     * @brief Vector of network layers
     * 
     * Public access is provided for advanced usage scenarios like reinforcement learning.
     * Generally, prefer using the provided interface methods instead of direct access.
     * Each Layer contains nodes and their weights, biases, and activation functions.
     * @warning Modifying layers directly may break network functionality unless you know what you're doing
     */
    std::vector<Layer<T>> m_layers;


protected:
    void UpdateWeights(const std::vector<std::vector<T>> & all_layers_activations,
                     const std::vector<T> &error,
                     float learning_rate);
    T _TrainOnExample(std::vector<T> feat, std::vector<T> label,
                      float learning_rate, T sampleSizeReciprocal);
    void CreateMLP(const std::vector<size_t> & layers_nodes,
                    const std::vector<ACTIVATION_FUNCTIONS> & layers_activfuncs,
                    loss::LOSS_FUNCTIONS loss_function = loss::LOSS_FUNCTIONS::LOSS_MSE,
                    bool use_constant_weight_init = false,
                    T constant_weight_init = 0.5);
    void ReportProgress(const bool output_log,
        const unsigned int every_n_iter,
        const unsigned int i,
        const T current_iteration_cost_function);
    void ReportFinish(const unsigned int i,
        const float current_iteration_cost_function);
    size_t m_num_inputs{ 0 };
    int m_num_outputs{ 0 };
    int m_num_hidden_layers{ 0 };
    std::vector<size_t> m_layers_nodes;
    MLP_LOSS_FN loss::loss_func_t<T> loss_fn_;
};

#endif //MLP_H
