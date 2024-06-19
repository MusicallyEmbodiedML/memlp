//============================================================================
// Name : Sample.h
// Author : David Nogueira
//============================================================================
#ifndef TRAININGSAMPLE_H
#define TRAININGSAMPLE_H

#include <iostream>
#include <stdlib.h>
#include <vector>

template<typename T>
class Sample {
public:
  Sample(const std::vector<T> & input_vector) {

    m_input_vector = input_vector;
  }
  const std::vector<T> & input_vector() const {
    return m_input_vector;
  }
  size_t GetInputVectorSize() const {
    return m_input_vector.size();
  }
  void AddBiasValue(T bias_value) {
    m_input_vector.insert(m_input_vector.begin(), bias_value);
  }
  friend std::ostream & operator<<(std::ostream &stream, Sample const & obj) {
    obj.PrintMyself(stream);
    return stream;
  };
protected:
  virtual void PrintMyself(std::ostream& stream) const {
    stream << "Input vector: [";
    for (size_t i = 0; i < m_input_vector.size(); i++) {
      if (i != 0)
        stream << ", ";
      stream << m_input_vector[i];
    }
    stream << "]";
  }

  std::vector<T> m_input_vector;
};


template<typename T>
class TrainingSample : public Sample<T> {
    using Sample<T>::m_input_vector;

public:
  TrainingSample(const std::vector<T> & input_vector,
                 const std::vector<T> & output_vector) :
    Sample<T>(input_vector) {
    m_output_vector = output_vector;
  }
  const std::vector<T> & output_vector() const {
    return m_output_vector;
  }
  size_t GetOutputVectorSize() const {
    return m_output_vector.size();
  }

protected:
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

  std::vector<T> m_output_vector;
};


#endif // TRAININGSAMPLE_H
