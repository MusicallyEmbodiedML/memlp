#ifndef __DATASET_HPP__
#define __DATASET_HPP__

#include <vector>
#include <cstdint>
#include <cstddef>

class Dataset {
 public:
    using DatasetVector = std::vector< std::vector<float> >;
    void Add(const std::vector<float> &feature, const std::vector<float> &label);
    void Train();
    void Clear();
    void Load(std::vector< std::vector<float> > &features,
              std::vector< std::vector<float> > &labels);
    void Fetch(std::vector< std::vector<float> > * &features,
               std::vector< std::vector<float> > * &labels);
    DatasetVector& GetFeatures();
    DatasetVector& GetLabels();

 protected:
    static constexpr unsigned int kMax_examples = 10;

    std::vector<std::vector<float>> features_;
    std::vector<std::vector<float>> labels_;
};

#endif  // __DATASET_HPP__
