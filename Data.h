/**
 * @file Data.h
 * @brief Common data type definitions for MEMLP Library
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef __DATA_H__
#define __DATA_H__

#ifdef __cplusplus

// C++ Includes
#include <vector>

// C and C++ symbols
extern "C" {
#endif

/**
 * @deprecated Use target architecture's single-precision float type instead.
 * This typedef is maintained for backwards compatibility only.
 * For new code, directly use 'float' which is guaranteed to be
 * single-precision (32-bit) on the target ARM architecture.
 */
typedef float num_t;

#ifdef __cplusplus
}

/**
 * @brief Convenience type alias for single-dimension vectors
 * @tparam T The data type of the vector elements
 */
template<typename T>
using d_vector = std::vector<T>;

/**
 * @brief Convenience type alias for nested (multi-dimensional) vectors
 * @tparam T The data type of the vector elements
 */
template<typename T>
using nd_vector = std::vector< d_vector<T> >;

#endif

#endif  // __DATA_H__