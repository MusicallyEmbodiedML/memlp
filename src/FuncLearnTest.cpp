#include "FuncLearnTest.hpp"
#include "Utils.h"
#if defined(MLP_VERBOSE)
#include <stdio.h>
#endif


FUNCLEARNTEST_C_FN
void groundtruth_fn(
    d_vector<number_t>& x,
    d_vector<number_t>& y)
{
    static const number_t x_shift = 1.f, y_shift = 0.f;

    assert(x.size() == y.size());
    
    // Iterate through the two vectors at the same time
    auto x_n = x.begin();
    auto y_n = y.begin();

    while (x_n != x.end() || y_n != y.end())
    {
        auto _x_n = *x_n;
        // Calculation body here
        _x_n -= x_shift;
        if (_x_n < -1.f) {
            *y_n = 1.f;
        } else if (_x_n > 1.f) {
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
    number_t epsilon = 0.00001f * utils::sgn<number_t>(stop);

    unsigned int counter = 0;
    for (number_t n = start; n < stop-epsilon; n += step) {
        out[counter] = n;
        counter++;
    }

    return out;
}


FuncLearnDataset::FuncLearnDataset(number_t lower_point,
                                   number_t upper_point,
                                   unsigned int training_points,
                                   unsigned int validation_points,
                                   FUNCLEARNTEST_C_FN
                                   groundtruth_fn_t groundtruth_fn_ptr) :
    training_points_(training_points),
    validation_points_(validation_points),
    groundtruth_fn_ptr_(groundtruth_fn_ptr)
{
    training_set_ = std::make_shared<pair_of_vectors>();
    validation_set_ = std::make_shared<pair_of_vectors>();

    auto temp_train_features = std::make_unique<d_vector<num_t>>();
    auto temp_train_labels = std::make_unique<d_vector<num_t>>();
    auto temp_valid_features = std::make_unique<d_vector<num_t>>();
    auto temp_valid_labels = std::make_unique<d_vector<num_t>>();

    *temp_train_features = arange(lower_point, upper_point,
        static_cast<number_t>(upper_point - lower_point) /
        static_cast<number_t>(training_points));
    temp_train_labels->resize(temp_train_features->size());
    groundtruth_fn_ptr(*temp_train_features, *temp_train_labels);

    *temp_valid_features = arange(lower_point, upper_point,
        static_cast<number_t>(upper_point - lower_point) /
        static_cast<number_t>(validation_points));
    temp_valid_labels->resize(temp_valid_features->size());
    groundtruth_fn_ptr(*temp_valid_features, *temp_valid_labels);

    // Reshape into (-1, 1)
    training_set_->first.resize(temp_train_features->size());
    training_set_->second.resize(temp_train_features->size());
    for(unsigned int n = 0; n < temp_train_features->size(); n++) {
        training_set_->first[n] = { (*temp_train_features)[n] };
        training_set_->second[n] = { (*temp_train_labels)[n] };
    }
    validation_set_->first.resize(temp_valid_features->size());
    validation_set_->second.resize(temp_valid_features->size());
    for(unsigned int n = 0; n < temp_valid_features->size(); n++) {
        validation_set_->first[n] = { (*temp_valid_features)[n] };
        validation_set_->second[n] = { (*temp_valid_labels)[n] };
    }
}


#if defined(__XS3A__)
#pragma stackfunction 10
#endif
void FUNCLEARNTEST_C_FN FuncLearnRunner::MakeData(const unsigned int n_examples)
{
    n_examples_ = n_examples;

    dataset_ = std::make_unique<FuncLearnDataset>(
        -5.f,
        5.f,
        n_examples,
        static_cast<unsigned int>(n_examples / 100. * 23.),
        &groundtruth_fn
    );

    training_set_ = dataset_->training();
    validation_set_ = dataset_->validation();

    // Xscope debug probes
    // TODO AM overload log vector to save ndarrays
    //probes_[0].log_vector(training_set_->first);
    //probes_[1].log_vector(training_set_->second);
    printf("\n");
}


void FuncLearnRunner::MakeModel()
{
    const std::vector<unsigned int> layers_nodes = {
        1, 4, 4, 1
    };
    const std::vector<std::string> layers_activfuncs = {
        "relu", "relu", "tanh"
    };
    bool use_constant_weight_init = false;
    number_t constant_weight_init = 0.5;
    mlp_ = std::make_unique< MLP<number_t> >(layers_nodes,
                                             layers_activfuncs,
                                             "mse",
                                             use_constant_weight_init,
                                             constant_weight_init);
}


void FuncLearnRunner::TrainModel(const unsigned int n_epochs)
{
    mlp_->Train(*training_set_, .001f, static_cast<int>(n_epochs), 0.f, true);
}


FUNCLEARNTEST_C_FN
void funclearntest_main()
{
    const unsigned int n_examples = 500;
    const unsigned int n_epochs = 500;
#if defined(MLP_VERBOSE)
    printf("-------------------------\n");
    printf("--- FuncLearnTest run ---\n");
    printf("-------------------------\n");
#endif

    FuncLearnRunner runner;
    runner.MakeData(n_examples);
    runner.MakeModel();
    runner.TrainModel(n_epochs);

#if defined(MLP_VERBOSE)
    printf("--- FuncLearnTest completed. ---\n");
#endif
}


#if defined(FUNCLEARN_MAIN)

int main(int argc, char* argv[])
{
    funclearntest_main();
    return 0;
}

#endif  // defined(FUNCLEARN_MAIN)
