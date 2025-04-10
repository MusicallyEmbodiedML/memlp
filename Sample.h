/**
 * @file Sample.h
 * @brief Sample and TrainingSample class definitions for MEMLP Library
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * This code is derived from David Alberto Nogueira's MLP project:
 * https://github.com/davidalbertonogueira/MLP
 */

#ifndef TRAININGSAMPLE_H
#define TRAININGSAMPLE_H

#include <stdlib.h>
#include <vector>

#if defined(MLP_DEBUG_BUILD)
#include <iostream>
#endif

/**
 * @brief Base class representing a sample with input features
 * 
 * @tparam T The data type of the input features (typically float)
 */
template<typename T>
class Sample {
public:
  /**
   * @brief Constructs a new Sample object
   * 
   * @param input_vector Vector containing the input features
   */
  Sample(const std::vector<T> & input_vector) {
    m_input_vector = input_vector;
  }

  /**
   * @brief Get the input vector
   * @return const reference to the input vector
   */
  const std::vector<T> & input_vector() const {
    return m_input_vector;
  }

  /**
   * @brief Get the size of the input vector
   * @return Size of input vector
   */
  size_t GetInputVectorSize() const {
    return m_input_vector.size();
  }

  /**
   * @brief Add a bias value to the beginning of input vector
   * @param bias_value The bias value to add
   */
  void AddBiasValue(T bias_value) {
    m_input_vector.insert(m_input_vector.begin(), bias_value);
  }

#if defined(MLP_DEBUG_BUILD)
  friend std::ostream & operator<<(std::ostream &stream, Sample const & obj) {
    obj.PrintMyself(stream);
    return stream;
  };
#endif

protected:
#if defined(MLP_DEBUG_BUILD)
  virtual void PrintMyself(std::ostream& stream) const {
    stream << "Input vector: [";
    for (size_t i = 0; i < m_input_vector.size(); i++) {
      if (i != 0)
        stream << ", ";
      stream << m_input_vector[i];
    }
    stream << "]";
  }
#endif

  std::vector<T> m_input_vector;
};

/**
 * @brief Class representing a training sample with both input features and expected outputs
 * 
 * Extends the base Sample class to include output/target values for training
 * 
 * @tparam T The data type of the input/output values (typically float)
 */
template<typename T>
class TrainingSample : public Sample<T> {
    using Sample<T>::m_input_vector;

public:
  /**
   * @brief Constructs a new Training Sample object
   * 
   * @param input_vector Vector containing the input features
   * @param output_vector Vector containing the expected outputs/targets
   */
  TrainingSample(const std::vector<T> & input_vector,
                 const std::vector<T> & output_vector) :
    Sample<T>(input_vector) {
    m_output_vector = output_vector;
  }

  /**
   * @brief Get the output vector
   * @return const reference to the output vector
   */
  const std::vector<T> & output_vector() const {
    return m_output_vector;
  }

  /**
   * @brief Get the size of the output vector
   * @return Size of output vector
   */
  size_t GetOutputVectorSize() const {
    return m_output_vector.size();
  }

protected:

#if defined(MLP_DEBUG_BUILD)
  virtual void PrintMyself(std::ostream& stream) const {
    stream << "Input vector: [";
    for (size_t i = 0; i < m_input_vector.size(); i++) {
      if (i != 0)
        stream << ", ";
      stream << m_input_vector[i];
    }
    stream << "]";

    stream << "; ";

    stream << "Output vector: [";
    for (size_t i = 0; i < m_output_vector.size(); i++) {
      if (i != 0)
        stream << ", ";
      stream << m_output_vector[i];
    }
    stream << "]";
  }
#endif

  std::vector<T> m_output_vector;
};

#endif // TRAININGSAMPLE_H
