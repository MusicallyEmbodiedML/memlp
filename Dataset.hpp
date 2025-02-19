#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <vector>
#include <cstdint>
#include <cstddef>

class Dataset {
 public:
    using DatasetVector = std::vector< std::vector<float> >;

    Dataset() { _InitSizes(); }
    Dataset(DatasetVector &features, DatasetVector &labels) :
         features_(features), labels_(labels) { _InitSizes(); _AdjustSizes(); }
    bool Add(const std::vector<float> &feature, const std::vector<float> &label);
    void Clear();
    void Load(DatasetVector &features,
              DatasetVector &labels);
    void Fetch(DatasetVector *features,
               DatasetVector *labels);
    DatasetVector GetFeatures(bool with_bias = true);
    DatasetVector& GetLabels();
    inline size_t GetFeatureSize(bool with_bias = true) { return data_size_ + with_bias; }
    inline size_t GetOutputSize() { return output_size_; }

 protected:
    static constexpr unsigned int kMax_examples = 100;
    size_t data_size_;
    size_t output_size_;

    inline void _InitSizes()  { data_size_ = 0; output_size_ = 0; }
    void _AdjustSizes();

    DatasetVector features_;
    DatasetVector labels_;
};

#endif  // __DATASET_HPP__
