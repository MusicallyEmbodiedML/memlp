#include "Loss.h"
#include "Utils.h"

#include <vector>
#include <memory>
#include "microunit.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP


using num_t = float;
using d_vector = std::vector<num_t>;
using nd_vector = std::vector< std::vector<num_t> >;


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


int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}
