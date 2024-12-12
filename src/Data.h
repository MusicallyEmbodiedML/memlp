#ifndef __DATA_H__
#define __DATA_H__

#ifdef __cplusplus

// C++ Includes
#include <vector>

// C and C++ symbols
extern "C" {
#endif


typedef float num_t;

#ifdef __cplusplus
}

// C++-only symbols
template<typename T>
using d_vector = std::vector<T>;

template<typename T>
using nd_vector = std::vector< d_vector<T> >;

#endif

#endif  // __DATA_H__