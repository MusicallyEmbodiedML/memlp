#include "Dataset.hpp"
#include <cstdio>
#include <cassert>


bool Dataset::Add(const std::vector<float> &feature, const std::vector<float> &label)
{
    if (data_size_ > 0) {
        assert(feature.size() == data_size_);
        assert(label.size() == output_size_);
    }
    if (features_.size() >= kMax_examples) {
        std::printf("MLP- Max dataset size of %d exceeded.\n", kMax_examples);
        return false;
    }

    auto feature_local = feature;
    auto label_local = label;
    features_.push_back(feature_local);
    labels_.push_back(label_local);
    std::printf("MLP- Added example.\n");
    std::printf("MLP- Feature size %d, label size %d.\n", features_.size(), labels_.size());
    _AdjustSizes();

    return true;
}

void Dataset::Clear()
{
    features_.clear();
    labels_.clear();

    _InitSizes();
}

void Dataset::Load(std::vector<std::vector<float>> &features,
                   std::vector<std::vector<float>> &labels)
{
    features_ = features;
    labels_ = labels;
}

void Dataset::Fetch(std::vector<std::vector<float>> *features,
                    std::vector<std::vector<float>> *labels)
{
    features = &features_;
    labels = &labels_;
}

Dataset::DatasetVector Dataset::GetFeatures(bool with_bias)
{
    DatasetVector features;
    features = features_;

    if (with_bias) {
        for (auto &f : features) {
            // Add bias
            f.push_back(1.f);
        }
    }

    return features;
}

Dataset::DatasetVector &Dataset::GetLabels()
{
    return labels_;
}

void Dataset::_AdjustSizes()
{
    if (features_.size() >= 1) {
        data_size_ = features_[0].size();
        output_size_ = labels_[0].size();
    }
}
