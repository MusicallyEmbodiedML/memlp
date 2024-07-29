//============================================================================
// Name : LayerTest.cpp
// Author : David Nogueira
//============================================================================

#include "Layer.h"
#include "Utils.h"

#include <vector>
#include <memory>
#include "microunit.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


using num_t = float;
using nd_vector = std::vector< std::vector<num_t> >;


/**
 * Compare output after activation function of first layer
 * from https://colab.research.google.com/drive/1DgNlTbtpo8XD8YEmGqpKu2ma4BiHRW_B
 */
UNIT(TestGetOutputAfterActivationFunction) {

    const unsigned int in_w = 1;
    const unsigned int in_b = 1;

    auto test_layer = std::make_unique< Layer<num_t> >(
        in_w+in_b, 4, "relu", false
    );

    std::vector< std::vector<num_t> > init_weights = {
        { -0.01876022294163704, -0.40200406312942505 },
        { 1.0124775171279907,   -1.3942354917526245 },
        { -0.9561545848846436,  0.2646740674972534 },
        { -1.1696271896362305,  0.801473081111908 }
    };

    test_layer->SetWeights(init_weights);

    nd_vector input = { { -1.0 }, { 0.0 }, { 1.0 } };
    nd_vector expected_output = {
        { -0.003832438262179494, -0.02406712993979454, 1.220828652381897, 1.9711003303527832 }
      , { -0.004020040389150381, -0.0139423543587327, 0.2646740674972534, 0.801473081111908 }
      , { -0.004207642748951912, -0.003817579708993435, -0.00691480515524745, -0.0036815409548580647 }
    };

    nd_vector actual_output;
    for (auto input_n : input) {
            std::vector<num_t> output_n;
            std::vector<num_t> input_n_with_bias {input_n[0], 1.};
            test_layer->GetOutputAfterActivationFunction(
                input_n_with_bias, &output_n
            );
            actual_output.push_back(output_n);
    }

    for (unsigned int n = 0; n < actual_output.size(); n++) {
        for (unsigned int k = 0; k < actual_output[n].size(); k++) {
            ASSERT_TRUE(utils::is_close<num_t>(expected_output[n][k], actual_output[n][k]));
        }
    }
}


int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}
