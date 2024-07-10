#ifndef _FUNC_LEARN_TEST_HPP_
#define _FUNC_LEARN_TEST_HPP_

#include <vector>
#include <cassert>
#include <utility>
#include <memory>

#include "MLP.h"

#define number_t    float

#if defined(__XS3A__)
#define FUNCLEARNTEST_C_FN __attribute__(( fptrgroup("funclearntest") ))
#else
#define FUNCLEARNTEST_C_FN
#endif

/**
 * Public functions
 */
FUNCLEARNTEST_C_FN
void groundtruth_fn(std::vector<number_t>& x, std::vector<number_t>& y);
std::vector<number_t> arange(number_t start, number_t stop, number_t step);

/**
 * Public definitions
 */

using pair_of_vectors = std::pair< std::vector<number_t>, std::vector<number_t> >;
using groundtruth_fn_t = void (*)(std::vector<number_t>&, std::vector<number_t>&);

/**
 * Public classes
 */

class FuncLearnDataset {

public:
    FuncLearnDataset(number_t lower_point,
        number_t upper_point,
        unsigned int training_points,
        unsigned int validation_points,
        groundtruth_fn_t groundtruth_fn_ptr);
    inline std::shared_ptr<pair_of_vectors> training() {
        return training_set_;
    }
    inline std::shared_ptr<pair_of_vectors> validation() {
        return validation_set_;
    }

protected:
    unsigned int training_points_;
    unsigned int validation_points_;
    groundtruth_fn_t groundtruth_fn_ptr_;
    std::shared_ptr<pair_of_vectors> training_set_;
    std::shared_ptr<pair_of_vectors> validation_set_;
};

class FuncLearnRunner {

public:

    void MakeData(const unsigned int n_examples);
    void MakeModel(void);
    void TrainModel(const unsigned int n_epochs);

protected:

    unsigned int n_examples_;
    std::shared_ptr<pair_of_vectors> training_set_;
    std::shared_ptr<pair_of_vectors> validation_set_;
    std::unique_ptr< MLP<number_t> > mlp_;
};


/**
 * C wrapper interface
 */

extern "C"
{
    FUNCLEARNTEST_C_FN
    void funclearntest_main(void);
}

#endif  // _FUNC_LEARN_TEST_HPP_
