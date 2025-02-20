#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <random>
#include <algorithm>

class Dataset {
 public:
    static constexpr size_t kMax_examples = 100;
    using DatasetVector = std::vector<std::vector<float>>;

    // Forget modes for replay memory.
    enum ForgetMode { FIFO, RANDOM_EQUAL, RANDOM_OLDER };

    Dataset();
    Dataset(DatasetVector &features, DatasetVector &labels);

    // Merged Add: if replay memory is enabled then, when full, an old example is removed.
    bool Add(const std::vector<float> &feature, const std::vector<float> &label);

    void Clear();
    void Load(DatasetVector &features, DatasetVector &labels);
    void Fetch(DatasetVector *&features, DatasetVector *&labels);

    /// @brief Returns a copy of the features, optionally adding a bias term.
    DatasetVector GetFeatures(bool with_bias = true);
    DatasetVector &GetLabels();
    inline size_t GetFeatureSize(bool with_bias = true) { return data_size_ + with_bias; }
    inline size_t GetOutputSize() { return output_size_; }

    // Toggle replay memory functionality on (true) or off (false).
    void ReplayMemory(bool enabled);

    // Set the forgetting mode for replay memory.
    void SetForgetMode(ForgetMode mode);

    // Set the maximum number of examples. If the new max is less than the current size,
    // extra examples are removed as follows:
    //   - If replay memory is enabled, repeatedly remove one excess example
    //     (using the same removal code as in Add()).
    //   - If disabled, simply trim the dataset (from the end).
    void SetMaxExamples(size_t max);

    // Sample returns a pair:
    //   first: feature vectors (optionally with bias appended),
    //   second: corresponding label vectors.
    // When replay memory is enabled, the ordering is randomized.
    // When disabled, the entire dataset is returned.
    std::pair<DatasetVector, DatasetVector> Sample(bool with_bias = true);

 protected:
    size_t data_size_;
    size_t output_size_;

    inline void _InitSizes() { data_size_ = 0; output_size_ = 0; }
    void _AdjustSizes();

    DatasetVector features_;
    DatasetVector labels_;

 private:
    // Utility static method that returns a copy of the given feature vectors,
    // adding a bias term (1.0f) to each vector if with_bias is true.
    static DatasetVector AddBias(const DatasetVector &features, bool with_bias);

    // Removes one excess example based on the current forget mode.
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
