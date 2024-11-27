#include "Dataset.hpp"
#include <cstdio>


void Dataset::Add(const std::vector<float> &feature, const std::vector<float> &label)
{
    if (features_.size() >= kMax_examples) {
        std::printf("MLP- Max dataset size of 10 exceeded.\n");
    } else {
        auto feature_local = feature;
        auto label_local = label;
        features_.push_back(feature_local);
        labels_.push_back(label_local);
        std::printf("MLP- Added example.\n");
        std::printf("MLP- Feature size %d, label size %d.\n", features_.size(), labels_.size());
    }
}

void Dataset::Clear()
{
    features_.clear();
    labels_.clear();
}

void Dataset::Load(std::vector<std::vector<float>> &features,
                   std::vector<std::vector<float>> &labels)
{
    features_ = features;
    labels_ = labels;
}

void Dataset::Fetch(const std::vector<std::vector<float>> *features,
                    const std::vector<std::vector<float>> *labels)
{
    features = &features_;
    labels = &labels_;
}

Dataset::DatasetVector &Dataset::GetFeatures()
{
    return features_;
}

Dataset::DatasetVector &Dataset::GetLabels()
{
    return labels_;
}
