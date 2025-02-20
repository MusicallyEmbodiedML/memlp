#include "Dataset.hpp"
#include <cstdio>
#include <cassert>
#include <random>
#include <algorithm>

Dataset::Dataset() : rng_(std::random_device{}()) {
    _InitSizes();
    max_examples_ = kMax_examples; // Default maximum.
}

Dataset::Dataset(DatasetVector &features, DatasetVector &labels)
    : features_(features), labels_(labels), rng_(std::random_device{}()) {
    _InitSizes();
    _AdjustSizes();
    max_examples_ = kMax_examples; // Default maximum.
    // Initialize timestamps for each loaded example.
    timestamps_.resize(features_.size());
    for (size_t i = 0; i < timestamps_.size(); i++) {
        timestamps_[i] = i;
    }
    current_timestamp_ = timestamps_.size();
}

bool Dataset::Add(const std::vector<float> &feature, const std::vector<float> &label)
{
    // Enforce consistent dimensions if at least one example exists.
    if (data_size_ > 0) {
        if ((feature.size() != data_size_) ||
            (label.size() != output_size_)) {
            std::printf("Dataset- Wrong example size.\n");
            return false;
        }
    }
    // When capacity is reached:
    if (features_.size() >= max_examples_) {
        if (replay_memory_enabled_) {
            RemoveOneExcessExample();
        } else {
            std::printf("Dataset- Max dataset size of %zu exceeded.\n", max_examples_);
            return false;
        }
    }
    // Add new example.
    features_.push_back(feature);
    labels_.push_back(label);
    timestamps_.push_back(current_timestamp_);
    current_timestamp_++;

    std::printf("Dataset- Added example.\n");
    std::printf("Dataset- Feature size %zu, label size %zu.\n", features_.size(), labels_.size());
    _AdjustSizes();
    return true;
}

void Dataset::RemoveOneExcessExample() {
    // Remove one example according to the current forget mode.
    size_t index_to_remove = 0;
    switch (forget_mode_) {
        case FIFO:
            index_to_remove = 0;
            break;
        case RANDOM_EQUAL:
        {
            std::uniform_int_distribution<size_t> dist(0, features_.size() - 1);
            index_to_remove = dist(rng_);
            break;
        }
        case RANDOM_OLDER:
        {
            size_t total_weight = 0;
            std::vector<size_t> weights;
            weights.reserve(timestamps_.size());
            for (size_t t : timestamps_) {
                size_t age = current_timestamp_ - t;
                weights.push_back(age);
                total_weight += age;
            }
            if (total_weight == 0) {
                std::uniform_int_distribution<size_t> dist(0, features_.size() - 1);
                index_to_remove = dist(rng_);
            } else {
                std::uniform_int_distribution<size_t> dist(0, total_weight - 1);
                size_t r = dist(rng_);
                size_t cumulative = 0;
                for (size_t i = 0; i < weights.size(); i++) {
                    cumulative += weights[i];
                    if (r < cumulative) {
                        index_to_remove = i;
                        break;
                    }
                }
            }
            break;
        }
        default:
            index_to_remove = 0;
            break;
    }
    // Remove the selected example from all parallel vectors.
    features_.erase(features_.begin() + index_to_remove);
    labels_.erase(labels_.begin() + index_to_remove);
    timestamps_.erase(timestamps_.begin() + index_to_remove);
    std::printf("Dataset- Memory full, removing example at index %zu (mode %d).\n", index_to_remove, forget_mode_);
}

void Dataset::Clear()
{
    features_.clear();
    labels_.clear();
    timestamps_.clear();
    current_timestamp_ = 0;
    _InitSizes();
}

void Dataset::Load(DatasetVector &features, DatasetVector &labels)
{
    features_ = features;
    labels_ = labels;
    _AdjustSizes();
    // Reinitialize timestamps for loaded examples.
    timestamps_.resize(features_.size());
    for (size_t i = 0; i < timestamps_.size(); i++) {
        timestamps_[i] = i;
    }
    current_timestamp_ = timestamps_.size();
}

void Dataset::Fetch(DatasetVector *&features, DatasetVector *&labels)
{
    features = &features_;
    labels = &labels_;
}

Dataset::DatasetVector Dataset::AddBias(const DatasetVector &features, bool with_bias)
{
    DatasetVector result = features; // make a copy
    if (with_bias) {
        for (auto &f : result) {
            f.push_back(1.f);
        }
    }
    return result;
}

Dataset::DatasetVector Dataset::GetFeatures(bool with_bias)
{
    return AddBias(features_, with_bias);
}

Dataset::DatasetVector &Dataset::GetLabels()
{
    return labels_;
}

void Dataset::_AdjustSizes()
{
    if (!features_.empty()) {
        data_size_ = features_[0].size();
        output_size_ = labels_[0].size();
    }
}

void Dataset::ReplayMemory(bool enabled)
{
    replay_memory_enabled_ = enabled;
    if (replay_memory_enabled_) {
        std::printf("Replay memory functionality enabled.\n");
    } else {
        std::printf("Replay memory functionality disabled.\n");
    }
}

void Dataset::SetForgetMode(ForgetMode mode)
{
    forget_mode_ = mode;
    std::printf("Forget mode set to %d.\n", mode);
}

void Dataset::SetMaxExamples(size_t max)
{
    max_examples_ = max;
    // If the current dataset size exceeds the new maximum, remove extra examples.
    while (features_.size() > max_examples_) {
        if (replay_memory_enabled_) {
            RemoveOneExcessExample();
        } else {
            // When replay memory is disabled, trim the extra examples from the end.
            features_.resize(max_examples_);
            labels_.resize(max_examples_);
            timestamps_.resize(max_examples_);
            break;
        }
    }
    std::printf("Max examples set to %zu.\n", max_examples_);
}

std::pair<Dataset::DatasetVector, Dataset::DatasetVector> Dataset::Sample(bool with_bias)
{
    std::pair<DatasetVector, DatasetVector> samplePair;
    size_t currentSize = features_.size();
    if (currentSize == 0) {
        return samplePair;
    }

    if (replay_memory_enabled_) {
        // Create a list of indices and shuffle them.
        std::vector<size_t> indices(currentSize);
        for (size_t i = 0; i < currentSize; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), rng_);

        DatasetVector sampledFeatures;
        DatasetVector sampledLabels;
        sampledFeatures.reserve(currentSize);
        sampledLabels.reserve(currentSize);

        for (size_t idx : indices) {
            sampledFeatures.push_back(features_[idx]);
            sampledLabels.push_back(labels_[idx]);
        }
        // Add bias if requested.
        samplePair.first = AddBias(sampledFeatures, with_bias);
        samplePair.second = sampledLabels;
    } else {
        // Replay memory disabled: return the entire dataset.
        samplePair.first = GetFeatures(with_bias);
        samplePair.second = labels_;
    }
    return samplePair;
}
