//============================================================================
// Name : Layer.h
// Author : David Nogueira
//============================================================================
#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <algorithm>
#include <cassert> // for assert()
#include "Node.h"
#include "Utils.h"


// Definition of activation function pointer
template<typename T>
using activation_func_t = T(*)(T);


template<typename T>
class Layer {
public:
  Layer() {
    m_num_nodes = 0;
    m_nodes.clear();
  };

  Layer(int num_inputs_per_node,
        int num_nodes,
        const std::string & activation_function,
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
    m_activation_function_str = activation_function;
  };

  ~Layer() {
    m_num_inputs_per_node = 0;
    m_num_nodes = 0;
    m_nodes.clear();
  };

  int GetInputSize() const {
    return m_num_inputs_per_node;
  };

  int GetOutputSize() const {
    return m_num_nodes;
  };

  const std::vector<Node<T>> & GetNodes() const {
    return m_nodes;
  }

  /**
   * Return the internal list of nodes, but modifiable.
   */
  std::vector<Node<T>> & GetNodesChangeable() {
    return m_nodes;
  }


  void GetOutputAfterActivationFunction(const std::vector<T> &input,
                                        std::vector<T> * output) {
    assert(input.size() == m_num_inputs_per_node);

    output->resize(m_num_nodes);

    for (size_t i = 0; i < m_num_nodes; ++i) {
      m_nodes[i].GetOutputAfterActivationFunction(input,
                                                  m_activation_function,
                                                  &((*output)[i]));
    }
  }

  void UpdateWeights(const std::vector<T> &input_layer_activation,
                     const std::vector<T> &deriv_error,
                     float m_learning_rate,
                     std::vector<T> * deltas) {
    assert(input_layer_activation.size() == m_num_inputs_per_node);
    assert(deriv_error.size() == m_nodes.size());

    deltas->resize(m_num_inputs_per_node, 0);

    for (size_t i = 0; i < m_nodes.size(); i++) {

      //dE/dwij = dE/doj . doj/dnetj . dnetj/dwij
      T dE_doj = 0;
      T doj_dnetj = 0;
      T dnetj_dwij = 0;

      dE_doj = deriv_error[i];
      doj_dnetj = m_deriv_activation_function(m_nodes[i].inner_prod); //cached from earlier calculation

      for (size_t j = 0; j < m_num_inputs_per_node; j++) {
        (*deltas)[j] += dE_doj * doj_dnetj * m_nodes[i].GetWeights()[j];

        dnetj_dwij = input_layer_activation[j];

        m_nodes[i].UpdateWeight(j,
                                static_cast<float>( -(dE_doj * doj_dnetj * dnetj_dwij) ),
                                m_learning_rate);
      }
    }
  };


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

  void SaveLayer(FILE * file) const {
    fwrite(&m_num_nodes, sizeof(m_num_nodes), 1, file);
    fwrite(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file);

    size_t str_size = m_activation_function_str.size();
    fwrite(&str_size, sizeof(size_t), 1, file);
    fwrite(m_activation_function_str.c_str(), sizeof(char), str_size, file);

    for (size_t i = 0; i < m_nodes.size(); i++) {
      m_nodes[i].SaveNode(file);
    }
  };
  
  void LoadLayer(FILE * file) {
    m_nodes.clear();

    fread(&m_num_nodes, sizeof(m_num_nodes), 1, file);
    fread(&m_num_inputs_per_node, sizeof(m_num_inputs_per_node), 1, file);

    size_t str_size = 0;
    fread(&str_size, sizeof(size_t), 1, file);
    m_activation_function_str.resize(str_size);
    fread(&(m_activation_function_str[0]), sizeof(char), str_size, file);

    std::pair<activation_func_t<T>,
        activation_func_t<T> > *pair;
    bool ret_val = utils::ActivationFunctionsManager<T>::Singleton().
      GetActivationFunctionPair(m_activation_function_str,
                                &pair);
    assert(ret_val);
    m_activation_function = (*pair).first;
    m_deriv_activation_function = (*pair).second;
    
    m_nodes.resize(m_num_nodes);
    for (size_t i = 0; i < m_nodes.size(); i++) {
      m_nodes[i].LoadNode(file);
    }

  };

  std::vector<Node<T>> m_nodes;

protected:
  size_t m_num_inputs_per_node{ 0 };
  size_t m_num_nodes{ 0 };

  std::string m_activation_function_str;
  MLP_ACTIVATION_FN activation_func_t<T> m_activation_function;
  MLP_ACTIVATION_FN activation_func_t<T> m_deriv_activation_function;
};

#endif //LAYER_H
