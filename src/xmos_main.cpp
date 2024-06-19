#include <iostream>
#include <xcore/hwtimer.h>
#include "MLP.h"


template<typename T>
void TestMLP() {
    // Test set up
    std::vector<TrainingSample<T>> training_set =
    {
        { { 0, 0 },{ 0.0 } },
        { { 0, 1 },{ 1.0 } },
        { { 1, 0 },{ 0.0 } },
        { { 1, 1 },{ 1.0 } }
    };
    bool bias_already_in = false;
    std::vector<TrainingSample<T>> training_sample_set_with_bias(training_set);
    //set up bias
    if (!bias_already_in) {
        for (auto & training_sample_with_bias : training_sample_set_with_bias) {
        training_sample_with_bias.AddBiasValue(1);
        }
    }

    size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
    size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
    MLP<T> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
    unsigned int epochs = 500;

    // Timer init
    auto timer = hwtimer_alloc();
    std::cout << "Benchmark starting (" << epochs << " epochs)..." << std::endl;
    auto now = hwtimer_get_time(timer);

    // Benchmarked code
    my_mlp.Train(training_sample_set_with_bias, 0.5, epochs, 0, false);

    // Timer benchmark
    auto t2 = hwtimer_get_time(timer);
	printf("start time: %lf\n", now * 1e-8);
	printf("end time: %lf\n", t2 * 1e-8);
	printf("total time: %lf\n", (t2-now) * 1e-8);
}


int main() {

    std::cout << "--- Test float32 MLP ---" << std::endl;
    TestMLP<float>();

    std::cout << "--- Test float64 MLP ---" << std::endl;
    TestMLP<double>();
}
