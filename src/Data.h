#ifndef __DATA_H__
#define __DATA_H__

#include <vector>

using num_t = float;

template<typename T>
using d_vector = std::vector<T>;

template<typename T>
using nd_vector = std::vector< d_vector<T> >;

#endif  // __DATA_H__