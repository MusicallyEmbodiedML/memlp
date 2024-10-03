//============================================================================
// Name : Node.h
// Author : David Nogueira
//============================================================================
#ifndef NODE_H
#define NODE_H

#include "Utils.h"

#include <vector>
#include <cassert> // for assert()
#include <numeric>
#include <algorithm>

#define CONSTANT_WEIGHT_INITIALIZATION 0


// Definition of activation function pointer
template<typename T>
using activation_func_t = T(*)(T);


template <typename T>
class Node {
public:
  Node() {
    m_num_inputs = 0;
    m_bias = 0;
    m_weights.clear();
  };
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

  int GetInputSize() const {
    return m_num_inputs;
  }

  void SetInputSize(int num_inputs) {
    m_num_inputs = num_inputs;
  }

  T GetBias() const {
    return m_bias;
  }
  void SetBias(T bias) {
    m_bias = bias;
  }

  std::vector<T> & GetWeights() {
    return m_weights;
  }

  const std::vector<T> & GetWeights() const {
    return m_weights;
  }

  void SetWeights( std::vector<T> & weights ){
      // check size of the weights vector
      assert(weights.size() == m_num_inputs);
      m_weights = weights;
  }

  size_t GetWeightsVectorSize() const {
    return m_weights.size();
  }

  T GetInputInnerProdWithWeights(const std::vector<T> &input) {
    
    static const T kInit(0);

    assert(input.size() == m_weights.size());
    T res = std::inner_product(begin(input),
                                           end(input),
                                           begin(m_weights),
                                           kInit);
    // *output = res;
    inner_prod = res;
    return inner_prod;
  }

  void GetOutputAfterActivationFunction(const std::vector<T> &input,
                                        MLP_ACTIVATION_FN activation_func_t<T> activation_function,
                                        T * output) {
    // T inner_prod = 0.0;
    GetInputInnerProdWithWeights(input);
    *output = activation_function(inner_prod);
  }

  void GetBooleanOutput(const std::vector<T> &input,
                        MLP_ACTIVATION_FN activation_func_t<T> activation_function,
                        bool * bool_output,
                        T threshold = 0.5) const {
    T value;
    GetOutputAfterActivationFunction(input, activation_function, &value);
    *bool_output = (value > threshold) ? true : false;
  };

  void UpdateWeights(const std::vector<T> &x,
                     T error,
                     T learning_rate) {
    assert(x.size() == m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
      m_weights[i] += x[i] * learning_rate *  error;
  };

  void UpdateWeight(int weight_id,
                    float increment,
                    float learning_rate) {
    m_weights[weight_id] += static_cast<T>(learning_rate*increment);
  }

  void SaveNode(FILE * file) const {
    fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
    fwrite(&m_bias, sizeof(m_bias), 1, file);
    fwrite(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file);
  };
  void LoadNode(FILE * file) {
    m_weights.clear();

    fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
    fread(&m_bias, sizeof(m_bias), 1, file);
    m_weights.resize(m_num_inputs);
    fread(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file);
  };
  
  std::vector<T> m_weights;
  T inner_prod;

  std::vector<T> inner_products;

  void prepareForOptimisation(size_t maxBatchSize) {
    inner_products.resize(maxBatchSize);
  }

protected:
  size_t m_num_inputs{ 0 };
  T m_bias{ 0.0 };
private:
};

#endif //NODE_H
