#ifndef __LOSS_H__
#define __LOSS_H__

#include <vector>
#include <cmath>
#include <unordered_map>
// #include <string>


#if defined(__XS3A__)

#define MLP_LOSS_FN __attribute__(( fptrgroup("mlp_loss") ))

#else

//#pragma message ( "PC compiler definitions enabled - check this is OK" )
#define MLP_LOSS_FN

#endif


namespace loss {


enum LOSS_FUNCTIONS {LOSS_MSE};



template<typename T>
MLP_LOSS_FN
inline T MSE(const std::vector<T> &expected, const std::vector<T> &actual,
             std::vector<T> &loss_deriv, T sampleSizeReciprocal) {
    
    T accum_loss = 0.;
    T n_elem = actual.size();
    T one_over_n_elem = 1. / n_elem;
    
    for (unsigned int j = 0; j < actual.size(); j++) {
        //TODO CK separate out diff for efficiency, replace pow with diff*diff
          accum_loss += std::pow((expected[j] - actual[j]), 2)
                * one_over_n_elem;
          loss_deriv[j] =
              -2 * one_over_n_elem
              * (expected[j] - actual[j]) * sampleSizeReciprocal;
    }
    accum_loss *= sampleSizeReciprocal;

    return accum_loss;
}


// Definition of loss function pointer
template<typename T>
using loss_func_t = T(*)(const std::vector<T> &, const std::vector<T> &, std::vector<T> &, T);


template<typename T>
class LossFunctionsManager {
 public:

    bool GetLossFunction(const LOSS_FUNCTIONS loss_name,
                         loss_func_t<T> *loss_fun) {
        
        auto iter = loss_functions_map.find(loss_name);
        if (iter != loss_functions_map.end()) {
            *loss_fun = iter->second;
        } else {
            return false;
        }
        return true;
    }

    static LossFunctionsManager & Singleton() {
        static LossFunctionsManager instance;
        return instance;
    }


 private:

    void AddNew(LOSS_FUNCTIONS function_name,
                loss_func_t<T> function) {
        loss_functions_map.insert(
            std::make_pair(function_name, function)
        );
    };

    LossFunctionsManager() {
        AddNew(LOSS_FUNCTIONS::LOSS_MSE, &MSE<T>);
    };

    std::unordered_map<
        LOSS_FUNCTIONS,
        loss_func_t<T>
    > loss_functions_map;
};


}  // namespace loss


#endif  // __LOSS_H__
