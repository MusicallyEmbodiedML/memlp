/**
 * @file Dataset.hpp
 * @brief Dataset management and replay memory functionality for MEMLP Library
 * @copyright Copyright (c) 2024. Licensed under Mozilla Public License Version 2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <random>
#include <algorithm>

/**
 * @brief Manages a dataset of feature-label pairs with optional replay memory functionality.
 * 
 * This class provides dataset management capabilities including loading, storing, and sampling
 * feature-label pairs. It includes legacy replay memory functionality which is now deprecated
 * in favor of the ReplayMemory class.
 */
class Dataset {
 public:
    static constexpr size_t kMax_examples = 100;
    using DatasetVector = std::vector<std::vector<float>>;

    /**
     * @brief Enumeration of forgetting modes for replay memory functionality.
     * @deprecated Use ReplayMemory::FORGETMODES instead.
     */
    enum ForgetMode { 
        FIFO,           /**< First-In-First-Out: Removes the oldest item. */
        RANDOM_EQUAL,   /**< Random Equal: Removes a random item with equal probability. */
        RANDOM_OLDER    /**< Random Older: Removes an older item with higher probability. */
    };

    /**
     * @brief Default constructor that initializes an empty dataset.
     */
    Dataset();

    /**
     * @brief Constructs a dataset with initial feature and label vectors.
     * 
     * @param features Vector of feature vectors to initialize with
     * @param labels Vector of label vectors to initialize with
     */
    Dataset(DatasetVector &features, DatasetVector &labels);

    /**
     * @brief Adds a new feature-label pair to the dataset.
     * 
     * @param feature Vector containing input features
     * @param label Vector containing output labels
     * @return true if addition was successful, false otherwise
     */
    bool Add(const std::vector<float> &feature, const std::vector<float> &label);

    /**
     * @brief Clears all data from the dataset.
     */
    void Clear();

    /**
     * @brief Loads feature and label vectors into the dataset.
     * 
     * @param features Vector of feature vectors to load
     * @param labels Vector of label vectors to load
     */
    void Load(DatasetVector &features, DatasetVector &labels);

    /**
     * @brief Provides direct access to internal feature and label vectors.
     * 
     * @param features Pointer to feature vectors will be stored here
     * @param labels Pointer to label vectors will be stored here
     */
    void Fetch(DatasetVector *&features, DatasetVector *&labels);

    /**
     * @brief Returns a copy of the feature vectors, optionally with bias terms.
     * 
     * @param with_bias If true, adds a bias term (1.0f) to each feature vector
     * @return DatasetVector Copy of feature vectors with optional bias terms
     */
    DatasetVector GetFeatures(bool with_bias = true);

    /**
     * @brief Returns a reference to the label vectors.
     * 
     * @return DatasetVector& Reference to the label vectors
     */
    DatasetVector &GetLabels();

    /**
     * @brief Returns the size of feature vectors, accounting for optional bias term.
     * 
     * @param with_bias If true, includes the bias term in the size
     * @return size_t Size of feature vectors
     */
    inline size_t GetFeatureSize(bool with_bias = true) { return data_size_ + with_bias; }

    /**
     * @brief Returns the size of label vectors.
     * 
     * @return size_t Size of label vectors
     */
    inline size_t GetOutputSize() { return output_size_; }

    /**
     * @brief Enables or disables replay memory functionality.
     * @deprecated Use ReplayMemory class instead.
     * 
     * @param enabled True to enable replay memory, false to disable
     */
    void ReplayMemory(bool enabled);

    /**
     * @brief Sets the forgetting mode for replay memory.
     * @deprecated Use ReplayMemory::FORGETMODES instead.
     * 
     * @param mode The forgetting mode to use
     */
    void SetForgetMode(ForgetMode mode);

    /**
     * @brief Sets the maximum number of examples in the dataset.
     * 
     * @param max Maximum number of examples to store
     */
    void SetMaxExamples(size_t max);

    /**
     * @brief Samples from the dataset, optionally with bias terms.
     * @deprecated Use ReplayMemory::sample() instead for replay memory functionality.
     * 
     * @param with_bias If true, adds bias terms to feature vectors
     * @return std::pair<DatasetVector, DatasetVector> Pair of feature and label vectors
     */
    std::pair<DatasetVector, DatasetVector> Sample(bool with_bias = true);

 protected:
    size_t data_size_;
    size_t output_size_;

    inline void _InitSizes() { data_size_ = 0; output_size_ = 0; }
    void _AdjustSizes();

    DatasetVector features_;
    DatasetVector labels_;

 private:
    /**
     * @brief Utility static method that returns a copy of the given feature vectors,
     * adding a bias term (1.0f) to each vector if with_bias is true.
     * 
     * @param features Feature vectors to copy
     * @param with_bias If true, adds a bias term to each vector
     * @return DatasetVector Copy of feature vectors with optional bias terms
     */
    static DatasetVector AddBias(const DatasetVector &features, bool with_bias);

    /**
     * @brief Removes one excess example based on the current forget mode.
     */
    void RemoveOneExcessExample();

    // Replay memory functionality flag.
    bool replay_memory_enabled_ = false;
    std::mt19937 rng_;

    // Additional members for extended replay memory functionality:
    // Timestamps for each example (used in RANDOM_OLDER mode).
    std::vector<size_t> timestamps_;
    size_t current_timestamp_ = 0;
    // Current forgetting mode.
    ForgetMode forget_mode_ = FIFO;

    // Maximum number of examples allowed.
    size_t max_examples_;
};

#endif  // __DATASET_HPP__
