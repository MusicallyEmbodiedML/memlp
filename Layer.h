/**
 * @file Layer.h
 * @brief Neural network layer implementation managing multiple nodes and their connections
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This code is derived from David Alberto Nogueira's MLP project:
 * https://github.com/davidalbertonogueira/MLP
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
#include <cassert> // for assert()
#include "Node.h"
#include "Utils.h"
#include <span>



/**
 * @brief Definition of activation function pointer
 * @tparam T The numeric type used for calculations
 */
template<typename T>
using activation_func_t = T(*)(T);

/**
 * @class Layer
 * @brief Represents a layer of neural network nodes with shared inputs and activation function
 * @tparam T The numeric type used for calculations (typically float or double)
 */
template<typename T>
class Layer {
public:
  /**
   * @brief Default constructor
   */
  Layer() {
    m_num_nodes = 0;
    m_nodes.clear();
  };

  /**
   * @brief Constructor with initialization parameters
   * @param num_inputs_per_node Number of inputs for each node in the layer
   * @param num_nodes Number of nodes in this layer
   * @param activation_function Activation function type for all nodes
   * @param use_constant_weight_init Flag to use constant weight initialization
   * @param constant_weight_init Value for constant weight initialization
   */
  Layer(int num_inputs_per_node,
        int num_nodes,
        const ACTIVATION_FUNCTIONS & activation_function,
        bool use_constant_weight_init = true,
        T constant_weight_init = 0.5) {
    m_num_inputs_per_node = num_inputs_per_node;
    m_num_nodes = num_nodes;
    m_nodes.resize(num_nodes);

    for (int i = 0; i < num_nodes; i++) {
      m_nodes[i].WeightInitialization(num_inputs_per_node,
                                      use_constant_weight_init,
                                      constant_weight_init);
    }

    std::pair<activation_func_t<T>,
        activation_func_t<T> > *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
      GetActivationFunctionPair(activation_function,
                                &pair);
    assert(ret_val);
    m_activation_function = (*pair).first;
    m_deriv_activation_function = (*pair).second;
    m_activation_function_type = activation_function;
  };

  /**
   * @brief Destructor
   */
  ~Layer() {
    m_num_inputs_per_node = 0;
    m_num_nodes = 0;
    m_nodes.clear();
  };

  /**
   * @brief Controls output caching behavior
   * @param onOrOff True to enable output caching, false to disable
   */
  void SetCachedOutputs(bool onOrOff) {
    m_cacheOutputs = onOrOff;
    if (m_cacheOutputs) {
      //nothing yet
    }else{
      cachedOutputs.clear();
    }
  }

  /**
   * @brief Gets the number of inputs per node
   * @return Number of inputs per node
   */
  int GetInputSize() const {
    return m_num_inputs_per_node;
  };

  /**
   * @brief Gets the number of nodes in the layer
   * @return Number of nodes in the layer
   */
  int GetOutputSize() const {
    return m_num_nodes;
  };

  /**
   * @brief Gets the list of nodes in the layer
   * @return Constant reference to the list of nodes
   */
  std::span<const Node<T>> GetNodes() {
    return m_nodes;
  }

  /**
   * @brief Return the internal list of nodes, but modifiable
   * @return Reference to the list of nodes
   */
  std::vector<Node<T>> & GetNodesChangeable() {
    return m_nodes;
  }

  /**
   * @brief Computes layer outputs using the activation function
   * @param input Input vector
   * @param output Pointer to store output vector
   */
  inline void GetOutputAfterActivationFunction(const std::vector<T> &input,
                                        std::vector<T> * output) {
    assert(input.size() == m_num_inputs_per_node);

    output->resize(m_num_nodes);

    for (size_t i = 0; i < m_num_nodes; ++i) {
      m_nodes[i].GetOutputAfterActivationFunction(input,
                                                  m_activation_function,
                                                  &((*output)[i]));
      // sleep_us(70);
    }
    if (m_cacheOutputs) {
      cachedOutputs = *output;
    }
  }

    /**
     * @brief Initialize gradient accumulators for all nodes
     */
    void InitializeGradientAccumulators() {
        for (auto& node : m_nodes) {
            node.InitializeGradientAccumulator();
        }
    }
    
    /**
     * @brief Clear gradient accumulators for all nodes
     */
    void ClearGradientAccumulators() {
        for (auto& node : m_nodes) {
            node.ClearGradientAccumulator();
        }
    }
    
    /**
     * @brief Accumulate gradients without updating weights (for batch training)
     * @param input_layer_activation Activation values of the input layer
     * @param deriv_error Derivative of the error with respect to outputs
     * @param deltas Pointer to store computed deltas for previous layer
     */
    void AccumulateGradients(const std::vector<T>& input_layer_activation,
                           const std::vector<T>& deriv_error,
                           std::vector<T>* deltas) {
        assert(input_layer_activation.size() == m_num_inputs_per_node);
        assert(deriv_error.size() == m_nodes.size());
        
        deltas->resize(m_num_inputs_per_node, 0);
        
        for (size_t i = 0; i < m_nodes.size(); i++) {
            T dE_doj = deriv_error[i];
            T doj_dnetj = m_deriv_activation_function(m_nodes[i].inner_prod);
            T error_signal = dE_doj * doj_dnetj;
            
            // Accumulate gradients in the node
            m_nodes[i].AccumulateGradients(input_layer_activation, error_signal);
            
            // Calculate deltas for previous layer
            for (size_t j = 0; j < m_num_inputs_per_node; j++) {
                (*deltas)[j] += error_signal * m_nodes[i].GetWeights()[j];
            }
        }
    }
    
    /**
     * @brief Apply accumulated gradients to all nodes
     * @param learning_rate Learning rate
     * @param batch_size Batch size for averaging
     */
    void ApplyAccumulatedGradients(float learning_rate, size_t batch_size) {
        for (auto& node : m_nodes) {
            node.ApplyAccumulatedGradients(learning_rate, batch_size);
        }
    }
        


  // /**
  //  * @brief Updates weights of the layer nodes
  //  * @param input_layer_activation Activation values of the input layer
  //  * @param deriv_error Derivative of the error with respect to outputs
  //  * @param m_learning_rate Learning rate for weight updates
  //  * @param deltas Pointer to store computed deltas
  //  */
  // void UpdateWeights(const std::vector<T> &input_layer_activation,
  //                    const std::vector<T> &deriv_error,
  //                    float m_learning_rate,
  //                    std::vector<T> * deltas) {
  //   assert(input_layer_activation.size() == m_num_inputs_per_node);
  //   assert(deriv_error.size() == m_nodes.size());

  //   deltas->resize(m_num_inputs_per_node, 0);

  //   for (size_t i = 0; i < m_nodes.size(); i++) {

  //     //dE/dwij = dE/doj . doj/dnetj . dnetj/dwij
  //     T dE_doj = 0;
  //     T doj_dnetj = 0;
  //     T dnetj_dwij = 0;

  //     dE_doj = deriv_error[i];
  //     doj_dnetj = m_deriv_activation_function(m_nodes[i].inner_prod); //cached from earlier calculation

  //     for (size_t j = 0; j < m_num_inputs_per_node; j++) {
  //       (*deltas)[j] += dE_doj * doj_dnetj * m_nodes[i].GetWeights()[j];

  //       dnetj_dwij = input_layer_activation[j];

  //       m_nodes[i].UpdateWeight(j,
  //                               static_cast<float>( -(dE_doj * doj_dnetj * dnetj_dwij) ),
  //                               m_learning_rate);
  //     }
  //   }
  // };

    /**
     * @brief Update weights with optional gradient accumulation
     * @param input_layer_activation Activation values
     * @param deriv_error Error derivatives
     * @param learning_rate Learning rate
     * @param deltas Computed deltas
     * @param accumulate If true, accumulate gradients instead of immediate update
     */
    void UpdateWeights(const std::vector<T>& input_layer_activation,
                      const std::vector<T>& deriv_error,
                      float learning_rate,
                      std::vector<T>* deltas,
                      bool accumulate = false) {
        
        if (accumulate) {
            AccumulateGradients(input_layer_activation, deriv_error, deltas);
        } else {
            // Original immediate update implementation
            assert(input_layer_activation.size() == m_num_inputs_per_node);
            assert(deriv_error.size() == m_nodes.size());
            
            deltas->resize(m_num_inputs_per_node, 0);
            
            for (size_t i = 0; i < m_nodes.size(); i++) {
                T dE_doj = deriv_error[i];
                T doj_dnetj = m_deriv_activation_function(m_nodes[i].inner_prod);
                
                for (size_t j = 0; j < m_num_inputs_per_node; j++) {
                    (*deltas)[j] += dE_doj * doj_dnetj * m_nodes[i].GetWeights()[j];
                    T dnetj_dwij = input_layer_activation[j];
                    m_nodes[i].UpdateWeight(j,
                                          static_cast<float>(-(dE_doj * doj_dnetj * dnetj_dwij)),
                                          learning_rate);
                }
            }
        }
    }
    
  

  /**
   * @brief Calculates gradients for optimization
   * @param input_layer_activation Activation values of the input layer
   * @param deriv_error Derivative of the error with respect to outputs
   * @param deltas Pointer to store computed deltas
   */
  void CalcGradients(const std::vector<T> &input_layer_activation,
                     const std::vector<T> &deriv_error,
                     std::vector<T> * deltas) {
    assert(input_layer_activation.size() == m_num_inputs_per_node);
    assert(deriv_error.size() == m_nodes.size());
    grads = deriv_error; //keep a copy
    deltas->resize(m_num_inputs_per_node, 0);
    for (size_t i = 0; i < m_nodes.size(); i++) {
      //dE/dwij = dE/doj . doj/dnetj . dnetj/dwij
      T dE_doj=0,doj_dnetj =0;
      dE_doj = deriv_error[i];
      doj_dnetj = m_deriv_activation_function(m_nodes[i].inner_prod); //cached from earlier calculation
      for (size_t j = 0; j < m_num_inputs_per_node; j++) {
        (*deltas)[j] += dE_doj * doj_dnetj * m_nodes[i].GetWeights()[j];
      }
    }
  };

  /**
   * @brief Sets gradients for optimization
   * @param newGrads New gradients to set
   */
  void SetGrads(std::vector<T> newGrads) {
    grads = newGrads;
  }

  /**
   * @brief Gets the stored gradients
   * @return Reference to the stored gradients
   */
  std::vector<T>& GetGrads() {
    return grads;
  }

  /**
   * @brief Sets weights for the layer nodes
   * @param weights 2D vector of weights for each node
   */
  void SetWeights( std::vector<std::vector<T>> & weights )
  {
      assert(0 <= weights.size() && weights.size() <= m_num_nodes /* Incorrect layer number in SetWeights call */);
      {
          // traverse the list of nodes
          size_t node_i = 0;
          for( Node<T> & node : m_nodes )
          {
              node.SetWeights( weights[node_i] );
              node_i++;
          }
      }
  };

  /**
   * @brief Smoothly updates weights using another layer's weights
   * @param l Reference to another layer
   * @param alpha Smoothing factor
   * @param alphaInv Inverse of smoothing factor
   */
  inline void SmoothUpdateWeights(Layer<T> &l, const float alpha, const float alphaInv) {
    // traverse the list of nodes
    for(size_t n=0; n < m_nodes.size(); n++) {
      m_nodes[n].SmoothUpdateWeights(l.m_nodes[n].GetWeights(), alpha, alphaInv);
      // sleep_us(70);
    }
  }

#if ENABLE_SAVE_SD
  bool SaveLayerSD(File &file) const {
    if (file.write((char*)&m_num_nodes, sizeof(m_num_nodes)) != sizeof(m_num_nodes)) {
      return false;
    }
    if (file.write((char*)&m_num_inputs_per_node, sizeof(m_num_inputs_per_node)) != sizeof(m_num_inputs_per_node)) {
      return false;
    }
    if (file.write((char*)&m_activation_function_type, sizeof(m_activation_function_type)) != sizeof(m_activation_function_type)) {
      return false;
    }

    for (size_t i = 0; i < m_nodes.size(); i++) {
      if (!m_nodes[i].SaveNodeSD(file)) {
        return false;
      }
    }
    return true;
  };

  bool LoadLayerSD(File &file) {
    m_nodes.clear();

    if (file.read((uint8_t*)&m_num_nodes, sizeof(m_num_nodes)) != sizeof(m_num_nodes)) {
        return false;
    }
    if (file.read((uint8_t*)&m_num_inputs_per_node, sizeof(m_num_inputs_per_node)) != sizeof(m_num_inputs_per_node)) {
        return false;
    }
    if (file.read((uint8_t*)&m_activation_function_type, sizeof(m_activation_function_type)) != sizeof(m_activation_function_type)) {
        return false;
    }

    std::pair<activation_func_t<T>,
        activation_func_t<T> > *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
      GetActivationFunctionPair(m_activation_function_type,
                                &pair);
    if (!ret_val) {
      return false;
    }
    m_activation_function = (*pair).first;
    m_deriv_activation_function = (*pair).second;

    m_nodes.resize(m_num_nodes);
    for (size_t i = 0; i < m_nodes.size(); i++) {
      if (!m_nodes[i].LoadNodeSD(file)) {
        return false;
      }
    }
    return true;
  };


#endif

#ifdef ENABLE_SAVE

  /**
   * @brief Saves the layer to a file
   * @param file File pointer to save the layer
   * @return true if save was successful, false if there was an error
   */
  bool SaveLayer(FILE * file) const {
    if (fwrite(&m_num_nodes, sizeof(m_num_nodes), 1, file) != 1) {
      return false;
    }
    if (fwrite(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file) != 1) {
      return false;
    }
    if (fwrite(&m_activation_function_type, sizeof(ACTIVATION_FUNCTIONS), 1, file) != 1) {
      return false;
    }

    for (size_t i = 0; i < m_nodes.size(); i++) {
      if (!m_nodes[i].SaveNode(file)) {
        return false;
      }
    }
    return true;
  };

  /**
   * @brief Loads the layer from a file
   * @param file File pointer to load the layer
   * @return true if load was successful, false if there was an error
   */
  bool LoadLayer(FILE * file) {
    m_nodes.clear();

    if (fread(&m_num_nodes, sizeof(m_num_nodes), 1, file) != 1) {
      return false;
    }
    if (fread(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file) != 1) {
      return false;
    }
    if (fread(&(m_activation_function_type), sizeof(ACTIVATION_FUNCTIONS), 1, file) != 1) {
      return false;
    }

    std::pair<activation_func_t<T>,
        activation_func_t<T> > *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
      GetActivationFunctionPair(m_activation_function_type,
                                &pair);
    if (!ret_val) {
      return false;
    }
    m_activation_function = (*pair).first;
    m_deriv_activation_function = (*pair).second;

    m_nodes.resize(m_num_nodes);
    for (size_t i = 0; i < m_nodes.size(); i++) {
      if (!m_nodes[i].LoadNode(file)) {
        return false;
      }
    }
    return true;
  };

#endif

  std::vector<Node<T>> m_nodes;

  std::vector<T> cachedOutputs;

  size_t m_num_inputs_per_node{ 0 }; /**< Number of inputs per node in this layer */
  size_t m_num_nodes{ 0 };           /**< Number of nodes in this layer */

protected:
  ACTIVATION_FUNCTIONS m_activation_function_type;              /**< Type of activation function used */
  activation_func_t<T> m_activation_function;                   /**< Pointer to activation function */
  activation_func_t<T> m_deriv_activation_function;             /**< Pointer to derivative of activation function */

  bool m_cacheOutputs{false};                                   /**< Flag controlling output caching */
  std::vector<T> grads;                                         /**< Stored gradients for optimization */
};

#endif //LAYER_H
