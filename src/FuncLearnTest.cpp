#include "easylogging++.h"
#include <vector>
#include <cassert>
#include <utility>
#include <memory>

#include "MLP.h"
#include "Utils.h"


#define number_t    float


number_t sgn(number_t val) {
    return (number_t(0) < val) - (val < number_t(0));
}


void groundtruth_fn(
    std::vector<number_t>& x,
    std::vector<number_t>& y)
{
    static const number_t x_shift = 1.f, y_shift = 0.f;

    assert(x.size == y.size);
    
    // Iterate through the two vectors at the same time
    auto x_n = x.begin();
    auto y_n = y.begin();

    while (x_n != x.end() || y_n != y.end())
    {
        auto _x_n = *x_n;
        // Calculation body here
        _x_n -= x_shift;
        if (_x_n < 1.f) {
            *y_n = 1.f;
        } else if (_x_n > 1) {
            *y_n = 1.f;
        } else {
            *y_n = _x_n * _x_n;
        }
        *y_n += y_shift;

        if (x_n != x.end())
        {
            ++x_n;
        }
        if (y_n != y.end())
        {
            ++y_n;
        }
    }

}


std::vector<number_t> arange(number_t start, number_t stop, number_t step = 1.f)
{
    int n_steps = static_cast<int>((stop - start) / step);
    assert(n_steps >= 0);
    std::vector<number_t> out(static_cast<unsigned int>(n_steps));

    // Mitigation for float rounding
    number_t epsilon = 0.00001 * sgn(stop);

    unsigned int counter = 0;
    for (number_t n = start; n < stop-epsilon; n += step) {
        out[counter] = n;
        counter++;
    }

    return out;
}


using pair_of_vectors = std::pair< std::vector<number_t>, std::vector<number_t> >;
using groundtruth_fn_t = void (*)(std::vector<number_t>&, std::vector<number_t>&);


class FuncLearnDataset {

public:
    FuncLearnDataset(number_t lower_point,
                     number_t upper_point,
                     unsigned int training_points,
                     unsigned int validation_points,
                     groundtruth_fn_t groundtruth_fn_ptr);
    std::shared_ptr<pair_of_vectors> training() {
        return training_set_;
    }
    std::shared_ptr<pair_of_vectors> validation() {
        return validation_set_;
    }

protected:
    unsigned int training_points_;
    unsigned int validation_points_;
    groundtruth_fn_t groundtruth_fn_ptr_;
    std::shared_ptr<pair_of_vectors> training_set_;
    std::shared_ptr<pair_of_vectors> validation_set_;
};


FuncLearnDataset::FuncLearnDataset(number_t lower_point,
                                   number_t upper_point,
                                   unsigned int training_points,
                                   unsigned int validation_points,
                                   groundtruth_fn_t groundtruth_fn_ptr) :
    training_points_(training_points),
    validation_points_(validation_points),
    groundtruth_fn_ptr_(groundtruth_fn_ptr)
{
    training_set_ = std::make_shared<pair_of_vectors>();
    validation_set_ = std::make_shared<pair_of_vectors>();

    training_set_->first = arange(lower_point, upper_point,
        static_cast<number_t>(upper_point - lower_point) /
        static_cast<number_t>(training_points));
    training_set_->second.resize(training_set_->first.size());
    groundtruth_fn_ptr(training_set_->first, training_set_->second);

    validation_set_->first = arange(lower_point, upper_point,
        static_cast<number_t>(upper_point - lower_point) /
        static_cast<number_t>(validation_points));
    validation_set_->second.resize(validation_set_->first.size());
    groundtruth_fn_ptr(validation_set_->first, validation_set_->second);
}


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


void FuncLearnRunner::MakeData(const unsigned int n_examples)
{
    n_examples_ = n_examples;

    auto dataset = std::make_unique<FuncLearnDataset>(
        -5.f,
        5.f,
        n_examples,
        static_cast<unsigned int>(n_examples / 100. * 23.),
        &groundtruth_fn
    );

    training_set_ = dataset->training();
    validation_set_ = dataset->validation();
}


void FuncLearnRunner::MakeModel()
{
    const std::vector<uint64_t> layers_nodes = {
        1, 4, 4, 1
    };
    const std::vector<std::string> layers_activfuncs = {
        "relu", "relu", "tanh"
    };
    bool use_constant_weight_init = false;
    number_t constant_weight_init = 0.5;
    mlp_ = std::make_unique< MLP<number_t> >(layers_nodes,
                                             layers_activfuncs,
                                             use_constant_weight_init,
                                             constant_weight_init);
}


void FuncLearnRunner::TrainModel(const unsigned int n_epochs)
{
    mlp_->Train(*training_set_, .001f, n_epochs, 0.10f, true);
}


int main(int argc, char* argv[]) {
    
    const unsigned int n_examples = 500;
    const unsigned int n_epochs = 500;
    
    FuncLearnRunner runner;
    runner.MakeData(n_examples);
    runner.MakeModel();
    runner.TrainModel(n_epochs);


    LOG(INFO) << "Test completed." << std::endl;
}
