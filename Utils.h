/**
 * @file Utils.h
 * @brief Utility functions and structures for machine learning operations
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This code is derived from David Alberto Nogueira's MLP project:
 * https://github.com/davidalbertonogueira/MLP
 */

#ifndef UTILS_H
#define UTILS_H

#include <unordered_map>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>

/**
 * @enum ACTIVATION_FUNCTIONS
 * @brief Enumeration of supported activation functions.
 */
enum ACTIVATION_FUNCTIONS {
    SIGMOID, /**< Sigmoid activation function */
    TANH,    /**< Hyperbolic tangent activation function */
    LINEAR,  /**< Linear activation function */
    RELU     /**< Rectified Linear Unit (ReLU) activation function */
};

#if defined(_WIN32) || (defined(__unix__) || defined(__unix) || (defined(__APPLE__) && defined(__MACH__)))
#include <cstdio>
#define ENABLE_SAVE    1
#endif

#ifdef ARDUINO
#define ENABLE_SAVE    1
#endif

#if defined(__XS3A__)
#define MLP_ACTIVATION_FN __attribute__(( fptrgroup("mlp_activation") ))
#else
//#pragma message ( "PC compiler definitions enabled - check this is OK" )
#define MLP_ACTIVATION_FN
#endif

/**
 * @namespace utils
 * @brief Contains utility functions and structures for machine learning.
 */
namespace utils {

/**
 * @brief Computes the sigmoid of a value.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The sigmoid of the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T sigmoid(T x) {
  return 1 / (1 + std::exp(-x));
}

/**
 * @brief Computes the derivative of the sigmoid function.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The derivative of the sigmoid function at the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_sigmoid(T x) {
  return sigmoid(x)*(1 - sigmoid(x));
}

/**
 * @brief Computes the hyperbolic tangent of a value.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The hyperbolic tangent of the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T hyperbolic_tan(T x) {
    return (std::tanh)(x);
}

/**
 * @brief Computes the derivative of the hyperbolic tangent function.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The derivative of the hyperbolic tangent function at the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_hyperbolic_tan(T x) {
  return 1 - (std::pow)(hyperbolic_tan(x), 2);
}

/**
 * @brief Computes the linear function of a value.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The linear function of the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T linear(T x) {
  return x;
}

/**
 * @brief Computes the derivative of the linear function.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The derivative of the linear function.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_linear(T x) {
    return static_cast<T>(1);
}

static const float kReLUSlope = 0.01f;

/**
 * @brief Computes the ReLU function of a value.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The ReLU function of the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T relu(T x) {
    return (x > 0) ? x : kReLUSlope * x;
}

/**
 * @brief Computes the derivative of the ReLU function.
 * @tparam T The type of the input value.
 * @param x The input value.
 * @return The derivative of the ReLU function.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T deriv_relu(T x) {
    return (x > 0) ? 1 : kReLUSlope;
}

/**
 * @brief Computes the sign of a value.
 * @tparam T The type of the input value.
 * @param val The input value.
 * @return The sign of the input value.
 */
template<typename T>
MLP_ACTIVATION_FN
inline T sgn(T val) {
    return static_cast<T>( (T(0) < val) - (val < T(0)) );
}

/**
 * @typedef activation_func_t
 * @brief Type definition for activation function pointers.
 * @tparam T The type of the input value.
 */
template<typename T>
using activation_func_t = T(*)(T);

/**
 * @struct ActivationFunctionsManager
 * @brief Manages activation functions and their derivatives.
 * @tparam T The type of the input value.
 */
template<typename T>
struct ActivationFunctionsManager {
  /**
   * @brief Retrieves the activation function pair for a given activation name.
   * @param activation_name The name of the activation function.
   * @param pair Pointer to the activation function pair.
   * @return True if the activation function pair is found, false otherwise.
   */
  bool GetActivationFunctionPair(const ACTIVATION_FUNCTIONS & activation_name,
                                    std::pair<activation_func_t<T>,
                                    activation_func_t<T>> **pair) {
    auto iter = activation_functions_map.find(activation_name);
    if (iter != activation_functions_map.end())
      *pair = &(iter->second);
    else
      return false;
    return true;
  }

  /**
   * @brief Retrieves the singleton instance of ActivationFunctionsManager.
   * @return The singleton instance.
   */
  static ActivationFunctionsManager & Singleton() {
    static ActivationFunctionsManager instance;
    return instance;
  }
private:
  /**
   * @brief Adds a new activation function pair to the manager.
   * @param function_name The name of the activation function.
   * @param function The activation function.
   * @param deriv_function The derivative of the activation function.
   */
  void AddNewPair(ACTIVATION_FUNCTIONS function_name,
                  activation_func_t<T> function,
                  activation_func_t<T> deriv_function) {
    activation_functions_map.insert(std::make_pair(function_name,
                                                   std::make_pair(function,
                                                                  deriv_function)));
  }

  /**
   * @brief Constructor for ActivationFunctionsManager.
   */
  ActivationFunctionsManager() {
    AddNewPair(ACTIVATION_FUNCTIONS::SIGMOID, &sigmoid<T>, &deriv_sigmoid<T>);
    AddNewPair(ACTIVATION_FUNCTIONS::TANH, &hyperbolic_tan<T>, &deriv_hyperbolic_tan<T>);
    AddNewPair(ACTIVATION_FUNCTIONS::LINEAR, &linear<T>, &deriv_linear<T>);
    AddNewPair(ACTIVATION_FUNCTIONS::RELU, &relu<T>, &deriv_relu<T>);
  }

  std::unordered_map<
    ACTIVATION_FUNCTIONS,
    std::pair< activation_func_t<T>, activation_func_t<T> >
  > activation_functions_map;
};

/**
 * @struct gen_rand
 * @brief Generates random numbers in a uniform distribution.
 * @tparam T The type of the generated random numbers.
 */
template<typename T>
struct gen_rand {
  T factor; /**< Scaling factor for random number generation. */
  T offset; /**< Offset for random number generation. */

  /**
   * @brief Constructor for gen_rand.
   * @param r The range of the random numbers.
   */
  gen_rand(T r = 2.0) : factor(r / static_cast<T>(RAND_MAX)), offset(r * 0.5) {}

  /**
   * @brief Generates a random number.
   * @return A random number in the range [-offset, offset].
   */
  T operator()() {
    return static_cast<T>(rand()) * factor - offset;
  }
};

/**
 * @struct gen_randn
 * @brief Generates random numbers in a normal distribution.
 * @tparam T The type of the generated random numbers.
 */
template<typename T>
struct gen_randn {
  T mean_; /**< Mean of the normal distribution. */
  T stddev_; /**< Standard deviation of the normal distribution. */
  gen_rand<T> gen_; /**< Uniform random number generator. */

  /**
   * @brief Constructor for gen_randn.
   * @param stddev The standard deviation of the normal distribution.
   * @param mean The mean of the normal distribution.
   */
  gen_randn(T stddev, T mean = 0) : mean_(mean), stddev_(stddev) {}

  /**
   * @brief Sets the mean of the normal distribution.
   * @param mean The mean to set.
   */
  inline void SetMean(T mean) { mean_ = mean; }

  /**
   * @brief Generates a random number with the current mean.
   * @return A random number in the normal distribution.
   */
  inline T operator()() {
    return operator()(mean_);
  }

  /**
   * @brief Generates a random number with a specified mean.
   * @param mean The mean to use for generation.
   * @return A random number in the normal distribution.
   */
  inline T operator()(T mean) {
    T accum = 0;
    static const unsigned int kN_times = 3;
    for (unsigned int n = 0; n < kN_times; n++) {
      accum += gen_();
    }
    return kN_times*(accum) * stddev_ + mean;
  }
};

/**
 * @brief Applies the softmax function to a vector.
 * @tparam T The type of the elements in the vector.
 * @param output Pointer to the vector to apply softmax to.
 */
template<typename T>
MLP_ACTIVATION_FN
inline void Softmax(std::vector<T> *output) {
  size_t num_elements = output->size();
  std::vector<T> exp_output(num_elements);
  T exp_total = 0;
  for (size_t i = 0; i < num_elements; i++) {
    float output_i = (*output)[i];
    if (output_i > 15.f) {
      output_i = 15.f;
    } else if (output_i < -15.f) {
      output_i = -15.f;
    }
    exp_output[i] = std::exp((*output)[i]);
    exp_total += exp_output[i];
  }
  for (size_t i = 0; i < num_elements; i++) {
    (*output)[i] = exp_output[i] / exp_total;
  }
}

/**
 * @brief Finds the index of the maximum element in a vector.
 * @tparam T The type of the elements in the vector.
 * @param output The vector to search.
 * @param class_id Pointer to store the index of the maximum element.
 */
template<typename T>
MLP_ACTIVATION_FN
inline void GetIdMaxElement(const std::vector<T> &output, size_t * class_id) {
  *class_id = std::distance(output.begin(),
                            std::max_element(output.begin(),
                                             output.end()));
}

/**
 * @brief Checks if two values are approximately equal.
 * @tparam T The type of the values.
 * @param a The first value.
 * @param b The second value.
 * @return True if the values are approximately equal, false otherwise.
 */
template<typename T>
inline bool is_close(T a, T b) {
    static const T kRelTolerance = 0.0001;
    a = std::abs(a);
    b = std::abs(b);
    T abs_tolerance = b*kRelTolerance;
    return (a < b + abs_tolerance) && (a > b - abs_tolerance);
}

}  // namespace utils

extern "C" int getentropy (void * buffer, size_t how_many);

#endif // UTILS_H
