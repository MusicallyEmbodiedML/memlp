//============================================================================
// Name : Utils.h
// Author : David Nogueira
//============================================================================
#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <numeric>
#include <chrono>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <typeindex>
#include <cassert>

#include "Chrono.h"
#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#endif

#if defined(__XS3A__)

#define MLP_ACTIVATION_FN __attribute__(( fptrgroup("mlp_activation") ))

#else

#pragma message ( "PC compiler definitions enabled - check this is OK" )
#define MLP_ACTIVATION_FN

#endif

namespace utils {
//Typical sigmoid function created from input x
//Returns the sigmoided value
template<typename T>
MLP_ACTIVATION_FN
inline T sigmoid(T x) {
  return 1 / (1 + std::exp(-x));
}

// Derivative of sigmoid function
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_sigmoid(T x) {
  return sigmoid(x)*(1 - sigmoid(x));
}

//Compute hyperbolic tangent (tanh)
//Returns the hyperbolic tangent of x.
template<typename T>
MLP_ACTIVATION_FN
inline T hyperbolic_tan(T x) {
    return (std::tanh)(x);
}

// Derivative of hyperbolic tangent function
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_hyperbolic_tan(T x) {
  return 1 - (std::pow)(hyperbolic_tan(x), 2);
}

// Linear function
template<typename T>
MLP_ACTIVATION_FN
inline T linear(T x) {
  return x;
}

// Derivative of linear function
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_linear(T x) {
    return static_cast<T>(1);
};


// ReLU function
template<typename T>
MLP_ACTIVATION_FN
inline T relu(T x) {
    return (x > 0) ? x : 0;
}


// Derivative of ReLU function
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_relu(T x) {
    return (x > 0) ? 1 : 0;
}


// Definition of activation function pointer
template<typename T>
using activation_func_t = T(*)(T);


template<typename T>
struct ActivationFunctionsManager {
  bool GetActivationFunctionPair(const std::string & activation_name,
                                    std::pair<activation_func_t<T>,
                                    activation_func_t<T>> **pair) {
    auto iter = activation_functions_map.find(activation_name);
    if (iter != activation_functions_map.end())
      *pair = &(iter->second);
    else
      return false;
    return true;
  }

  static ActivationFunctionsManager & Singleton() {
    static ActivationFunctionsManager instance;
    return instance;
  }
private:
  void AddNewPair(std::string function_name,
                  activation_func_t<T> function,
                  activation_func_t<T> deriv_function) {
    activation_functions_map.insert(std::make_pair(function_name,
                                                   std::make_pair(function,
                                                                  deriv_function)));
  };

  ActivationFunctionsManager() {
    AddNewPair("sigmoid", &sigmoid<T>, &deriv_sigmoid<T>);
    AddNewPair("tanh", &hyperbolic_tan<T>, &deriv_hyperbolic_tan<T>);
    AddNewPair("linear", &linear<T>, &deriv_linear<T>);
    AddNewPair("relu", &relu<T>, &deriv_relu<T>);
  };

  std::unordered_map<
    std::string,
    std::pair< activation_func_t<T>, activation_func_t<T> >
  > activation_functions_map;
};

// TODO AM handle loss functions != MSE here

template<typename T>
struct gen_rand {
  T factor;
  T offset;
public:
  gen_rand(T r = 2.0) : factor(r / RAND_MAX), offset(r / 2) {}
  T operator()() {
    return rand() * factor - offset;
  }
};


template<typename T>
MLP_ACTIVATION_FN
inline void Softmax(std::vector<T> *output) {
  size_t num_elements = output->size();
  std::vector<T> exp_output(num_elements);
  T exp_total = 0;
  for (size_t i = 0; i < num_elements; i++) {
    exp_output[i] = std::exp((*output)[i]);
    exp_total += exp_output[i];
  }
  for (size_t i = 0; i < num_elements; i++) {
    (*output)[i] = exp_output[i] / exp_total;
  }
}


template<typename T>
MLP_ACTIVATION_FN
inline void GetIdMaxElement(const std::vector<T> &output, size_t * class_id) {
  *class_id = std::distance(output.begin(),
                            std::max_element(output.begin(),
                                             output.end()));
}

}  // namespace utils

#endif // UTILS_H
