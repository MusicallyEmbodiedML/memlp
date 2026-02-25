//============================================================================
// Name : MLP.cpp
//============================================================================
#include "MLP.h"


#if !defined(ARDUINO)

#include <stdio.h>

#else  // ARDUINO includes

#include <Arduino.h>
//#define PICO_BENCHMARK
#include <VFS.h>
#include <LittleFS.h>
#define ENABLE_SAVE    1

#endif

#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "utils/Serialise.hpp"
#include <cassert>
#include <random>

// #define SAFE_MODE


//desired call syntax :  MLP({64*64,20,4}, {"sigmoid", "linear"},
template<typename T>
MLP<T>::MLP(const std::vector<size_t> & layers_nodes,
         const std::vector<ACTIVATION_FUNCTIONS> & layers_activfuncs,
         loss::LOSS_FUNCTIONS loss_function,
         bool use_constant_weight_init,
         T constant_weight_init) : g(rd()) {
#ifdef SAFE_MODE
  assert(layers_nodes.size() >= 2);
  assert(layers_activfuncs.size() + 1 == layers_nodes.size());
#endif

  CreateMLP(layers_nodes,
            layers_activfuncs,
            loss_function,
            use_constant_weight_init,
            constant_weight_init);
};


#if ENABLE_SAVE
template<typename T>
MLP<T>::MLP(const std::string & filename) {
  if (!LoadMLPNetwork(filename)) {
    // If loading fails, we need to have a valid but empty network
    // Initialize with minimal valid configuration
    m_num_inputs = 0;
    m_num_outputs = 0;
    m_num_hidden_layers = 0;
    m_layers_nodes.clear();
    m_layers.clear();
    // Consider throwing an exception or setting an error flag here
    // For now, we'll have an invalid network that should be checked
  }
}
#endif


template<typename T>
MLP<T>::~MLP() {
  m_num_inputs = 0;
  m_num_outputs = 0;
  m_num_hidden_layers = 0;
  m_layers_nodes.clear();
  m_layers.clear();
};


template<typename T>
void MLP<T>::CreateMLP(const std::vector<size_t> & layers_nodes,
                    const std::vector<ACTIVATION_FUNCTIONS> & layers_activfuncs,
                    loss::LOSS_FUNCTIONS loss_function,
                    bool use_constant_weight_init,
                    T constant_weight_init) {
    m_layers_nodes = layers_nodes;
    m_num_inputs = m_layers_nodes[0];
    m_num_outputs = m_layers_nodes[m_layers_nodes.size() - 1];
    m_num_hidden_layers = m_layers_nodes.size() - 2;

    // Store loss function type for inference decisions
    m_loss_function_type = loss_function;

    // Loss function selection
    loss::LossFunctionsManager<T> loss_mgr =
        loss::LossFunctionsManager<T>::Singleton();
    assert(loss_mgr.GetLossFunction(loss_function, &(this->loss_fn_)));

    for (size_t i = 0; i < m_layers_nodes.size() - 1; i++) {
        m_layers.emplace_back(Layer<T>(m_layers_nodes[i],
                                    m_layers_nodes[i + 1],
                                    layers_activfuncs[i],
                                    use_constant_weight_init,
                                    constant_weight_init));
    }

    AllocateBuffers();
}


template<typename T>
void MLP<T>::AllocateBuffers() {
    size_t max_layer_size = *std::max_element(m_layers_nodes.begin(),
                                               m_layers_nodes.end());
    // Forward pass double buffers
    m_fwd_buf_a.resize(max_layer_size);
    m_fwd_buf_b.resize(max_layer_size);
    // Backprop double buffers
    m_bp_error.resize(max_layer_size);
    m_bp_deltas.resize(max_layer_size);
    // Training scratch
    m_train_predicted.resize(m_num_outputs);
    m_train_deriv_error.resize(m_num_outputs);
}


template<typename T>
void MLP<T>::ReportProgress(const bool output_log,
    const unsigned int every_n_iter,
    const unsigned int i,
    const T sampleLoss)
{
#if !defined(ARDUINO)
    if (output_log && ((i % every_n_iter) == 0)) {
        printf("Iteration %i cost function f(error): %f\n",
                i, sampleLoss);
    }
#endif
}


template<typename T>
void MLP<T>::ReportFinish(const unsigned int i, const float current_iteration_cost_function)
{
#if !defined(ARDUINO)
    printf("Iteration %i cost function f(error): %f\n",
        i, current_iteration_cost_function);

    printf("******************************\n");
    printf("******* TRAINING ENDED *******\n");
    printf("******* %d iters *******\n", i);
    printf("******************************\n");;
#endif
};


#if ENABLE_SAVE

template<typename T>
bool MLP<T>::SaveMLPNetwork(const std::string & filename) const {
    FILE * file = fopen(filename.c_str(), "wb");
    if (!file) {
        return false;
    }

    // Write network structure
    if (fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file) != 1) {
        fclose(file);
        return false;
    }
    if (fwrite(&m_num_outputs, sizeof(m_num_outputs), 1, file) != 1) {
        fclose(file);
        return false;
    }
    if (fwrite(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file) != 1) {
        fclose(file);
        return false;
    }

    // Write layer nodes
    if (!m_layers_nodes.empty()) {
        if (fwrite(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file) != m_layers_nodes.size()) {
            fclose(file);
            return false;
        }
    }

    // Write layer weights
    for (size_t i = 0; i < m_layers.size(); i++) {
        if (!m_layers[i].SaveLayer(file)) {
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

template<typename T>
bool MLP<T>::LoadMLPNetwork(const std::string & filename) {
    // Check if file exists
    FILE * file = fopen(filename.c_str(), "rb");
    if (!file) {
        return false;
    }

    // Clear existing network
    m_layers_nodes.clear();
    m_layers.clear();

    // Read network structure
    if (fread(&m_num_inputs, sizeof(m_num_inputs), 1, file) != 1) {
        fclose(file);
        return false;
    }
    if (fread(&m_num_outputs, sizeof(m_num_outputs), 1, file) != 1) {
        fclose(file);
        return false;
    }
    if (fread(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file) != 1) {
        fclose(file);
        return false;
    }

    // Read layer nodes
    m_layers_nodes.resize(m_num_hidden_layers + 2);
    if (!m_layers_nodes.empty()) {
        if (fread(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file) != m_layers_nodes.size()) {
            fclose(file);
            return false;
        }
    }

    // Read layer weights
    m_layers.resize(m_layers_nodes.size() - 1);
    for (size_t i = 0; i < m_layers.size(); i++) {
        if (!m_layers[i].LoadLayer(file)) {
            fclose(file);
            return false;
        }
    }

    fclose(file);
    AllocateBuffers();
    return true;
}

#endif

#if ENABLE_SAVE_SD
template<typename T>
bool MLP<T>::SaveMLPNetworkSD(const std::string & filename) {
        auto file = SD.open(filename.c_str(), FILE_WRITE);
        if (!file) {
            Serial.println("Failed to open file for writing");
            return false;
        }
        file.seek(0);  // Start from beginning
        // Write network structure
        if (file.write((const char*)&m_num_inputs, sizeof(m_num_inputs)) != sizeof(m_num_inputs)) {
            file.close();
            return false;
        }
        if (file.write((const char*)&m_num_outputs, sizeof(m_num_outputs)) != sizeof(m_num_outputs)) {
            file.close();
            return false;
        }
        if (file.write((const char*)&m_num_hidden_layers, sizeof(m_num_hidden_layers)) != sizeof(m_num_hidden_layers)) {
            file.close();
            return false;
        }
        // Write layer nodes
        if (!m_layers_nodes.empty()) {
            int dataSize = m_layers_nodes.size() * sizeof(m_layers_nodes[0]);
            if (file.write((char*)&m_layers_nodes[0], dataSize) != dataSize) {
                file.close();
                return false;
            }
        }

        // Write layer weights
        for (size_t i = 0; i < m_layers.size(); i++) {
            if (!m_layers[i].SaveLayerSD(file)) {
                file.close();
                return false;
            }
        }

        file.close();
        return true;
    }

 template<typename T>
bool MLP<T>::LoadMLPNetworkSD(const std::string & filename) {
    // Check if file exists
    auto file = SD.open(filename.c_str(), FILE_READ);
    if (!file) {
        return false;
    }

    // Clear existing network
    m_layers_nodes.clear();
    m_layers.clear();

    // Read network structure
    if (file.read((uint8_t*)&m_num_inputs, sizeof(m_num_inputs)) != sizeof(m_num_inputs)) {
        file.close();
        return false;
    }
    if (file.read((uint8_t*)&m_num_outputs, sizeof(m_num_inputs)) != sizeof(m_num_outputs)) {
        file.close();
        return false;
    }
    if (file.read((uint8_t*)&m_num_hidden_layers, sizeof(m_num_inputs)) != sizeof(m_num_hidden_layers)) {
        file.close();
        return false;
    }

    // Read layer nodes
    m_layers_nodes.resize(m_num_hidden_layers + 2);

    if (!m_layers_nodes.empty()) {
        int dataSize = m_layers_nodes.size() * sizeof(T);
        if (file.read((uint8_t*)&m_layers_nodes[0], dataSize) != dataSize) {
            return false;
        }
    }

    // Read layer weights
    m_layers.resize(m_layers_nodes.size() - 1);
    for (size_t i = 0; i < m_layers.size(); i++) {
        if (!m_layers[i].LoadLayerSD(file)) {
            file.close();
            return false;
        }
    }

    file.close();
    AllocateBuffers();
    return true;
}

#endif



template <typename T>
size_t MLP<T>::Serialise(size_t w_head, std::vector<uint8_t> &buffer)
{
    for (unsigned int n = 0; n < m_layers.size(); n++) {
        auto layer_weights = GetLayerWeights(n);
        w_head = Serialise::FromVector2D(w_head, layer_weights, buffer);
    }
    return w_head;
}


template <typename T>
size_t MLP<T>::FromSerialised(size_t r_head, const std::vector<uint8_t> &buffer)
{
    for (unsigned int n = 0; n < m_layers.size(); n++) {
        std::vector< std::vector<T> > layer_weights;
        r_head = Serialise::ToVector2D(r_head, buffer, layer_weights);
        SetLayerWeights(n, layer_weights);
    }
    return r_head;
};


// ─── CHANGED: GetOutput ─── zero-alloc inference via pre-allocated double buffers
template<typename T>
void MLP<T>::GetOutput(const std::vector<T> &input,
                    std::vector<T> * output,
                    std::vector<std::vector<T>> * all_layers_activations,
                    bool for_inference) {
#ifdef ARDUINO
    if (input.size() != m_num_inputs) {
        Serial.printf("ERROR: input.size()=%d != m_num_inputs=%d\n", input.size(), m_num_inputs);
        return;
    }
#else
    assert(input.size() == m_num_inputs);
#endif

    std::vector<T> *in_ptr  = &m_fwd_buf_a;
    std::vector<T> *out_ptr = &m_fwd_buf_b;

    // resize is a no-op: buffers already >= max layer size
    in_ptr->resize(input.size());
    std::copy(input.begin(), input.end(), in_ptr->begin());

    for (size_t i = 0; i < m_layers.size(); ++i) {
        out_ptr->resize(m_layers[i].GetOutputSize());

        m_layers[i].GetOutputAfterActivationFunction(*in_ptr, out_ptr);

        if (all_layers_activations != nullptr) {
            // Training path: must copy because we need all activations retained
            all_layers_activations->emplace_back(*in_ptr);
        }

        std::swap(in_ptr, out_ptr);
    }

    // After the loop, in_ptr holds the final output
    if (for_inference &&
        m_loss_function_type == loss::LOSS_FUNCTIONS::LOSS_CATEGORICAL_CROSSENTROPY &&
        in_ptr->size() > 1) {
        utils::Softmax(in_ptr);
    }

    // One copy out to caller
    output->resize(in_ptr->size());
    std::copy(in_ptr->begin(), in_ptr->end(), output->begin());
}


template<typename T>
void MLP<T>::GetOutputClass(const std::vector<T> &output, size_t * class_id) const {
    utils::GetIdMaxElement(output, class_id);
}


// ─── CHANGED: UpdateWeights ─── uses pre-allocated bp buffers, swap pattern
template<typename T>
void MLP<T>::UpdateWeights(const std::vector<std::vector<T>> & all_layers_activations,
                        const std::vector<T> &deriv_error,
                        float learning_rate) {

    std::vector<T> *err_ptr   = &m_bp_error;
    std::vector<T> *delta_ptr = &m_bp_deltas;

    err_ptr->resize(deriv_error.size());
    std::copy(deriv_error.begin(), deriv_error.end(), err_ptr->begin());

    for (int i = m_num_hidden_layers; i >= 0; --i) {
        delta_ptr->assign(m_layers[i].GetInputSize(), T(0));
        m_layers[i].UpdateWeights(all_layers_activations[i], *err_ptr, learning_rate, delta_ptr);
        if (i > 0) {
            std::swap(err_ptr, delta_ptr);
        }
    }
};


// ─── CHANGED: TrainBatch ─── uses member buffers, no per-batch allocations
template<typename T>
T MLP<T>::TrainBatch(const training_pair_t& training_sample_set,
                     float learning_rate,
                     int max_iterations,
                     size_t batch_size,
                     float min_error_cost,
                     bool output_log) {
    
    const auto& training_features = training_sample_set.first;
    const auto& training_labels = training_sample_set.second;
    
    size_t n_samples = training_features.size();
    size_t n_batches = (n_samples + batch_size - 1) / batch_size;

    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Reusable per-sample vectors — allocated once, cleared per sample
    std::vector<std::vector<T>> all_layers_activations;

    T epoch_loss = 0;
    for (int iter = 0; iter < max_iterations; iter++) {

        epoch_loss = 0;
        
        std::shuffle(indices.begin(), indices.end(), g);
        
        size_t sample_idx = 0;
        
        for (size_t batch = 0; batch < n_batches; batch++) {
            size_t current_batch_size = std::min(batch_size, n_samples - sample_idx);
            T batch_size_reciprocal = (T)1.0 / static_cast<T>(current_batch_size);
            
            InitializeAllGradientAccumulators();
            
            T batch_loss = 0;

            for (size_t i = 0; i < current_batch_size; i++) {
                size_t idx = indices[sample_idx++];
#ifdef SAFE_MODE
                if (idx >= training_features.size()) {
                    Serial.printf("ERROR: idx %d >= training_features.size() %d\n", idx, training_features.size());
                    continue;
                }
#endif
                // Forward pass — GetOutput uses m_fwd_buf_a/b internally
                all_layers_activations.clear();
                GetOutput(training_features[idx],
                         &m_train_predicted,
                         &all_layers_activations,
                         false);

                // Compute loss — reuse m_train_deriv_error
                m_train_deriv_error.resize(m_train_predicted.size());
                T loss = loss_fn_(training_labels[idx],
                                 m_train_predicted,
                                 m_train_deriv_error,
                                 1.0f);

                #ifdef MLP_ALLOW_DEBUG
                if (std::isinf(loss) || std::isnan(loss)) {
                    Serial.printf("[MLP DEBUG] *** INF/NAN loss at sample %d! loss=%f\n", i, loss);
                    Serial.printf("[MLP DEBUG]   pred[0]=%f, label[0]=%f\n",
                                 m_train_predicted[0], training_labels[idx][0]);
                }
                #endif

                batch_loss += loss;

                // Backprop — uses m_bp_error/m_bp_deltas internally
                BackpropagateWithAccumulation(all_layers_activations,
                                             m_train_deriv_error,
                                             true);
            }

            // Gradient clipping
            T grad_sumsq = 0.0f;
            for (auto& layer : m_layers) {
                grad_sumsq += layer.GetGradSumSquared(batch_size_reciprocal);
            }
            T grad_norm = std::sqrt(grad_sumsq);

            #ifdef MLP_ALLOW_DEBUG
            Serial.printf("[MLP DEBUG] Batch %d/%d: batch_loss=%f, grad_norm=%f\n",
                         batch, n_batches, batch_loss / current_batch_size, grad_norm);
            if (std::isinf(grad_norm) || std::isnan(grad_norm)) {
                Serial.printf("[MLP DEBUG] *** INF/NAN grad_norm! ***\n");
            }
            #endif

            if (grad_norm > 5.0f) {
                T clip_coef = 5.0f / grad_norm;
                for (auto& layer : m_layers) {
                    layer.ScaleAccumulatedGradients(clip_coef);
                }
            }

            ApplyAllAccumulatedGradients(learning_rate, batch_size_reciprocal);

            epoch_loss += batch_loss / current_batch_size;
        }

        epoch_loss /= n_batches;

        #ifdef MLP_ALLOW_DEBUG
        if (std::isinf(epoch_loss) || std::isnan(epoch_loss)) {
            Serial.printf("[MLP DEBUG] *** INF/NAN epoch_loss after iteration %d! ***\n", iter);
        }
        #endif

        if (output_log && (iter % 100 == 0)) {
            ReportProgress(output_log, 100, iter, epoch_loss);
        }

        if (m_progress_callback) {
            m_progress_callback(iter, epoch_loss);
        }

        if (epoch_loss < min_error_cost) {
            break;
        }
    }

    #ifdef MLP_ALLOW_DEBUG
    Serial.printf("[MLP DEBUG] TrainBatch returning epoch_loss=%f (inf=%d, nan=%d)\n",
                 epoch_loss, std::isinf(epoch_loss), std::isnan(epoch_loss));
    #endif

    return epoch_loss;
}


// ─── CHANGED: BackpropagateWithAccumulation ─── uses pre-allocated bp buffers
template<typename T>
void MLP<T>::BackpropagateWithAccumulation(const std::vector<std::vector<T>>& all_layers_activations,
                                           const std::vector<T>& deriv_error,
                                           bool accumulate) {
    std::vector<T> *err_ptr   = &m_bp_error;
    std::vector<T> *delta_ptr = &m_bp_deltas;

    err_ptr->resize(deriv_error.size());
    std::copy(deriv_error.begin(), deriv_error.end(), err_ptr->begin());

    for (int i = m_num_hidden_layers; i >= 0; --i) {
        delta_ptr->assign(m_layers[i].GetInputSize(), T(0));
        m_layers[i].UpdateWeights(all_layers_activations[i],
                                 *err_ptr,
                                 0,  // Learning rate not used when accumulating
                                 delta_ptr,
                                 accumulate);

        if (i > 0) {
            std::swap(err_ptr, delta_ptr);
        }
    }
}


// ─── CHANGED: Train (legacy) ─── const refs moved outside loop
template<typename T>
T MLP<T>::Train(const training_pair_t& training_sample_set_with_bias,
    float learning_rate,
    int max_iterations,
    float min_error_cost,
    bool) {

    int i = 0;
    T current_iteration_cost_function = 0.f;

    const auto& training_features = training_sample_set_with_bias.first;
    const auto& training_labels = training_sample_set_with_bias.second;
    T sampleSizeReciprocal = 1.f / training_features.size();

    for (i = 0; i < max_iterations; i++) {
        current_iteration_cost_function = 0.f;

        auto t_feat = training_features.begin();
        auto t_label = training_labels.begin();

        while (t_feat != training_features.end() || t_label != training_labels.end()) {

            current_iteration_cost_function +=
                _TrainOnExample(*t_feat, *t_label, learning_rate, sampleSizeReciprocal);

            if (t_feat != training_features.end())
            {
                ++t_feat;
            }
            if (t_label != training_labels.end())
            {
                ++t_label;
            }
        }

        current_iteration_cost_function *= sampleSizeReciprocal;

#if !defined(ARDUINO)
        ReportProgress(true, 100, i, current_iteration_cost_function);
#endif

        if (m_progress_callback && !(i & 0x1F)) {
            m_progress_callback(i, current_iteration_cost_function);
        }

        if (current_iteration_cost_function < min_error_cost) {
            break;
        }

    }

#if !defined(ARDUINO)
    ReportFinish(i, current_iteration_cost_function);
#else
     MLP_DEBUG_PRINT("### Training ended, iteration ");
     MLP_DEBUG_PRINT(i);
     MLP_DEBUG_PRINT(", loss ");
     MLP_DEBUG_PRINTLN(current_iteration_cost_function, 10);
#endif
    if (m_progress_callback) {
        m_progress_callback(i, current_iteration_cost_function);
    }

    return current_iteration_cost_function;
};


// ─── CHANGED: CalcGradients ─── uses pre-allocated bp buffers
template <typename T>
void MLP<T>::CalcGradients(std::vector<T> & feat, std::vector<T> & deriv_error_output)
{
    std::vector< std::vector<T> > all_layers_activations;

    GetOutput(feat,
        &m_train_predicted,
        &all_layers_activations,
        false);

    std::vector<T> *err_ptr   = &m_bp_error;
    std::vector<T> *delta_ptr = &m_bp_deltas;

    err_ptr->resize(deriv_error_output.size());
    std::copy(deriv_error_output.begin(), deriv_error_output.end(), err_ptr->begin());

    for (int i = m_num_hidden_layers; i >= 0; --i) {
        delta_ptr->assign(m_layers[i].GetInputSize(), T(0));
        m_layers[i].CalcGradients(all_layers_activations[i], *err_ptr, delta_ptr);
        if (i > 0) {
            std::swap(err_ptr, delta_ptr);
        } else {
            m_layers[0].SetGrads(*delta_ptr);
        }
    }
}


// ─── CHANGED: _TrainOnExample ─── uses member buffers
template <typename T>
T MLP<T>::_TrainOnExample(std::vector<T> feat,
                          std::vector<T> label,
                          float learning_rate,
                          T sampleSizeReciprocal)
{
    std::vector<std::vector<T>> all_layers_activations;

    GetOutput(feat,
        &m_train_predicted,
        &all_layers_activations,
        false);

    assert(label.size() == m_train_predicted.size());
    m_train_deriv_error.resize(m_train_predicted.size());

    T current_iteration_cost_function =
        this->loss_fn_(label, m_train_predicted,
        m_train_deriv_error, sampleSizeReciprocal);

    UpdateWeights(all_layers_activations,
        m_train_deriv_error,
        learning_rate);

    return current_iteration_cost_function;
}


// ─── CHANGED: ApplyLoss ─── uses member buffers
template <typename T>
void MLP<T>::ApplyLoss(std::vector<T> feat,
                          std::vector<T> loss,
                          float learning_rate)
{
    std::vector<std::vector<T>> all_layers_activations;

    GetOutput(feat,
        &m_train_predicted,
        &all_layers_activations,
        false);

    assert(loss.size() == m_train_predicted.size());

    UpdateWeights(all_layers_activations,
        loss,
        learning_rate);
}


// ─── CHANGED: AccumulatePolicyGradient ─── reuses m_train_deriv_error
template<typename T>
void MLP<T>::AccumulatePolicyGradient(const std::vector<T>& state,
                                  const std::vector<T>& action_gradient) {
    std::vector<std::vector<T>> all_layers_activations;
    
    GetOutput(state, &m_train_predicted, &all_layers_activations, false);
    
    // Negate gradients for maximization — reuse member buffer
    m_train_deriv_error.resize(action_gradient.size());
    for(size_t i = 0; i < action_gradient.size(); i++) {
        m_train_deriv_error[i] = -action_gradient[i];
    }
    
    BackpropagateWithAccumulation(all_layers_activations,
                                    m_train_deriv_error,
                                    true);
}


template <typename T>
void MLP<T>::Train(const std::vector<TrainingSample<T>>
        &training_sample_set_with_bias,
        float learning_rate,
        int max_iterations,
        float min_error_cost,
        bool output_log)
{
    std::vector< std::vector<T> > features, labels;

    for (const auto &sample : training_sample_set_with_bias) {
        features.push_back(sample.input_vector());
        labels.push_back(sample.output_vector());
    }

    training_pair_t t_pair(features, labels);
    Train(t_pair, learning_rate, max_iterations,
            min_error_cost, output_log);
};

template<typename T>
size_t MLP<T>::GetNumLayers()
{
    return m_layers.size();
}


// ─── CHANGED: GetLayerWeights ─── const ref instead of copying entire layer
template<typename T>
std::vector<std::vector<T>> MLP<T>::GetLayerWeights( size_t layer_i )
{
    assert(layer_i < m_layers.size());
    return m_layers[layer_i].GetWeights2D();
}

template <typename T>
typename MLP<T>::mlp_weights MLP<T>::GetWeights()
{
    MLP<T>::mlp_weights out;

    out.resize(m_layers.size());
    for (unsigned int n = 0; n < m_layers.size(); n++) {
        size_t num_nodes = m_layers[n].m_num_nodes;
        size_t num_inputs = m_layers[n].m_num_inputs_per_node;
        out[n].resize(num_nodes);
        for (unsigned int k = 0; k < num_nodes; k++) {
            out[n][k].resize(num_inputs);
            const T *row = m_layers[n].m_weights.data() + k * num_inputs;
            std::copy(row, row + num_inputs, out[n][k].begin());
        }
    }

    return out;
}

template<typename T>
void MLP<T>::SetLayerWeights( size_t layer_i, std::vector<std::vector<T>> & weights )
{
    assert(layer_i < m_layers.size());
    {
        m_layers[layer_i].SetWeights( weights );
    }
}

template <typename T>
void MLP<T>::SetWeights(MLP<T>::mlp_weights &weights)
{
#ifdef SAFE_MODE
    #if !defined(ARDUINO)
        if (weights.size() != m_layers.size()) {
            printf("SetWeights: vector dim not equal. Expected=%zu, actual=%zu",
                    weights.size(), m_layers.size());
        }
    #endif
    assert(weights.size() == m_layers.size());
#endif
#if 0
    for (unsigned int n = 0; n < m_layers.size(); n++) {
        size_t expected = m_layers[n].m_num_nodes;
        size_t actual = weights[n].size();
        bool assertion = expected == actual;
#if !defined(ARDUINO)
        if (!assertion) {
            printf("SetWeights: vector dim not equal at n=%zu. Expected=%zu, actual=%zu",
                   n, expected, actual);
        }
#endif
        assert(assertion);
        for (unsigned int k = 0; k < m_layers[n].m_num_nodes; k++) {
            size_t expected = m_layers[n].m_num_inputs_per_node;
            size_t actual = weights[n][k].size();
            bool assertion = expected == actual;
#if !defined(ARDUINO)
            if (!assertion) {
                printf("SetWeights: vector dim not equal at n=%d,k=%d. Expected=%d, actual=%d",
                    n, k, expected, actual);
            }
#endif
            assert(assertion);
        }
    }
#endif

    for (unsigned int n = 0; n < m_layers.size(); n++) {
        SetLayerWeights(n, weights[n]);
    }
}

template <typename T>
void MLP<T>::DrawWeights(float scale)
{
    utils::gen_rand<T> gen;

    for (unsigned int n = 0; n < m_layers.size(); n++) {
        for (T& w : m_layers[n].m_weights) {
            w = gen() * scale;
        }
    }
}

template <typename T>
void MLP<T>::MoveWeights(T speed)
{
    T before = m_layers[0].m_weights[0];
    utils::gen_randn<T> gen(speed);

    for (unsigned int n = 0; n < m_layers.size(); n++) {
        for (T& w : m_layers[n].m_weights) {
            T w_old = w;
            w = gen(w);
            if (speed != 0) {
                assert(w_old != w);
            }
        }
    }

    assert(m_layers[0].m_weights[0] != before);
}

template <typename T>
void MLP<T>::InitXavier() {
    for(auto & layer : m_layers) {
        layer.InitXavier();
    }
}

template <typename T>
void MLP<T>::RandomiseWeightsAndBiasesLin(T weightMin, T weightMax, T biasMin, T biasMax) {
    std::uniform_real_distribution<> disWeight(weightMin, weightMax);
    std::uniform_real_distribution<> disBias(biasMin, biasMax);

    for (unsigned int n = 0; n < m_layers.size(); n++) {
        for (T& w : m_layers[n].m_weights) {
            w = disWeight(g);
        }
        for (T& b : m_layers[n].m_biases) {
            b = disBias(g);
        }
    }
}


// Explicit instantiations
// #if !defined(__XS3A__)
// template class MLP<double>;
// #endif
template class MLP<float>;