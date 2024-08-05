#include <vector>
#include <memory>

#include "UnitTest.hpp"
#include "Loss.h"
#include "Utils.h"


UNIT(TestMSE) {
    const d_vector expected {1., 2., 3., 4.};
    const d_vector actual {-1., -2., -3., -4.};
    const d_vector expected_deriv {-1., -2., -3., -4. };
    d_vector deriv_out {0, 0, 0, 0};

    num_t loss = loss::MSE<num_t>(expected, actual, deriv_out);
    
    ASSERT_TRUE(utils::is_close<num_t>(loss, 30.));
    for (unsigned int n = 0; n < deriv_out.size(); n++) {
        ASSERT_TRUE(utils::is_close<num_t>(deriv_out[n], expected_deriv[n]));
    }
}


UNIT(TestLossFunctionsManager) {
    auto loss_mgr = loss::LossFunctionsManager<num_t>::Singleton();

    MLP_LOSS_FN loss::loss_func_t<num_t> loss_func;

    bool function_found = loss_mgr.GetLossFunction("mse", &loss_func);
    ASSERT_TRUE(function_found);
    bool found_right_function = loss_func == &loss::MSE<num_t>;
    ASSERT_TRUE(found_right_function);
    // Assert I can call the function without throwing
    std::vector<num_t> loss_deriv{0};
    loss_func({0}, {0}, loss_deriv);

    function_found = loss_mgr.GetLossFunction("invalid", &loss_func);
    ASSERT_FALSE(function_found);
}


#if defined(LOSSTEST_MAIN)

int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}

#endif  // LOSSTEST_MAIN