//============================================================================
// Name : MLP.cpp
// Author : David Nogueira
//============================================================================
#ifndef MLP_H
#define MLP_H

#include "Layer.h"
#include "Utils.h"
#include "Loss.h"
#include "Sample.h"

#include <cstdint>
#include <memory>
#include <string>



// template<typename T>
// class temptest {
//   public:
//   void test() {
//     // Serial.println("template test");
//   }
// };

template<typename T>
class MLP {
public:

    using training_pair_t = std::pair<
      std::vector< std::vector<T> >,
      std::vector< std::vector<T> >
    >;

    using mlp_weights = std::vector< std::vector <std::vector<T> > >;

    MLP(const std::vector<size_t> & layers_nodes,
        const std::vector<ACTIVATION_FUNCTIONS> & layers_activfuncs,
        loss::LOSS_FUNCTIONS loss_function = loss::LOSS_FUNCTIONS::LOSS_MSE,
        bool use_constant_weight_init = false,
        T constant_weight_init = 0.5);

#if ENABLE_SAVE
    MLP(const std::string & filename);
#endif
    ~MLP();

#if ENABLE_SAVE
    void SaveMLPNetwork(const std::string & filename) const;
    void LoadMLPNetwork(const std::string & filename);
#endif

    size_t Serialise(size_t w_head, std::vector<uint8_t> &buffer);
    size_t FromSerialised(size_t w_head, const std::vector<uint8_t> &buffer);

    void GetOutput(const std::vector<T> &input,
                    std::vector<T> * output,
                    std::vector<std::vector<T>> * all_layers_activations = nullptr);
    void GetOutputClass(const std::vector<T> &output, size_t * class_id) const;

    T Train(const training_pair_t& training_sample_set_with_bias,
                float learning_rate,
                int max_iterations = 5000,
                float min_error_cost = 0.001,
                bool output_log = true);

    T MiniBatchTrain(const training_pair_t& training_sample_set_with_bias,
                float learning_rate,
                int max_iterations = 5000,
                size_t miniBatchSize=8,
                float min_error_cost = 0.001,
                bool output_log = true);

    void Train(const std::vector<TrainingSample<T>> &training_sample_set_with_bias,
                        float learning_rate,
                        int max_iterations = 5000,
                        float min_error_cost = 0.001,
                        bool output_log = true);

    size_t GetNumLayers();
    std::vector<std::vector<T>> GetLayerWeights( size_t layer_i );
    mlp_weights GetWeights();
    void SetLayerWeights( size_t layer_i, std::vector<std::vector<T>> & weights );
    void SetWeights(mlp_weights &weights);
    void DrawWeights();
    void MoveWeights(T speed);

    std::vector<Layer<T>> m_layers;

    void SetCachedLayerOutputs(bool on) {
        for(auto &layer : m_layers) {
            layer.SetCachedOutputs(on);
        }
    }

    //used for target networks in RL
    void SmoothUpdateWeights(std::shared_ptr<MLP<T>> anotherMLP, const float alpha) {
        //assuming the other MLP has the same structure
        //calc this once here
        float alphaInv = 1.f-alpha;

        for(size_t i=0; i < m_layers.size(); i++) {
            m_layers[i].SmoothUpdateWeights(anotherMLP->m_layers[i], alpha, alphaInv);
        }
    }

    void CalcGradients(std::vector<T> &feat, std::vector<T> & deriv_error_output);

    void ClearGradients() {
        for(auto &v: m_layers) {
            v.SetGrads({});
        }
    }

protected:
    void UpdateWeights(const std::vector<std::vector<T>> & all_layers_activations,
                     const std::vector<T> &error,
                     float learning_rate);
    T _TrainOnExample(std::vector<T> feat, std::vector<T> label,
                      float learning_rate, T sampleSizeReciprocal);
    void CreateMLP(const std::vector<size_t> & layers_nodes,
                    const std::vector<ACTIVATION_FUNCTIONS> & layers_activfuncs,
                    loss::LOSS_FUNCTIONS loss_function = loss::LOSS_FUNCTIONS::LOSS_MSE,
                    bool use_constant_weight_init = false,
                    T constant_weight_init = 0.5);
    void ReportProgress(const bool output_log,
        const unsigned int every_n_iter,
        const unsigned int i,
        const T current_iteration_cost_function);
    void ReportFinish(const unsigned int i,
        const float current_iteration_cost_function);
    size_t m_num_inputs{ 0 };
    int m_num_outputs{ 0 };
    int m_num_hidden_layers{ 0 };
    std::vector<size_t> m_layers_nodes;
    MLP_LOSS_FN loss::loss_func_t<T> loss_fn_;
};

#endif //MLP_H
