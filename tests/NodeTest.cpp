//============================================================================
// Name : NodeTest.cpp
// Author : David Nogueira
//============================================================================
#include <vector>
#include <algorithm>

#include "UnitTest.hpp"
#include "Node.h"
#include "Sample.h"
#include "Utils.h"


namespace {
void Train(Node<num_t> & node,
           const std::vector<TrainingSample<num_t>> &training_sample_set_with_bias,
           double learning_rate,
           int max_iterations,
           bool use_constant_weight_init = true,
           double constant_weight_init = 0.5) {

  //initialize weight vector
  node.WeightInitialization(static_cast<int>(training_sample_set_with_bias[0].GetInputVectorSize()),
                            use_constant_weight_init,
                            constant_weight_init);

  //std::cout << "Starting weights:\t";
  //for (auto m_weightselement : node.GetWeights())
  //  std::cout << m_weightselement << "\t";
  //std::cout << std::endl;

  for (int i = 0; i < max_iterations; i++) {
    int error_count = 0;
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      bool prediction;
      node.GetBooleanOutput(training_sample_with_bias.input_vector(),
                            utils::linear<num_t>,
                            &prediction, 
                            0.5);
      bool correct_output = training_sample_with_bias.output_vector()[0] > 0.5 ? true : false;
      if (prediction != correct_output) {
        error_count++;
        double error = (correct_output ? 1 : 0) - (prediction ? 1 : 0);
        node.UpdateWeights(training_sample_with_bias.input_vector(),
                           learning_rate,
                           error);
      }
    }
    if (error_count == 0) break;
  }

  //std::cout << "Final weights:\t\t";
  //for (auto m_weightselement : node.GetWeights())
  //  std::cout << m_weightselement << "\t";
  //std::cout << std::endl;
};
}

UNIT(NodeLearnAND) {
  LOG(INFO) << "Train AND function with Node." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{0.0} },
    { { 0, 1 },{0.0} },
    { { 1, 0 },{0.0} },
    { { 1, 1 },{1.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node<num_t> my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(),
                             utils::linear<num_t>,
                             &class_id, 
                             0.5);
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  LOG(INFO) << "Trained with success." << std::endl;  
}

UNIT(NodeLearnNAND) {
  LOG(INFO) << "Train NAND function with Node." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{1.0} },
    { { 0, 1 },{1.0} },
    { { 1, 0 },{1.0} },
    { { 1, 1 },{0.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node<num_t> my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(),
                             utils::linear<num_t>,
                             &class_id,
                             0.5);
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  LOG(INFO) << "Trained with success." << std::endl;  
}

UNIT(NodeLearnOR) {
  LOG(INFO) << "Train OR function with Node." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{0.0} },
    { { 0, 1 },{1.0} },
    { { 1, 0 },{1.0} },
    { { 1, 1 },{1.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node<num_t> my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(),
                             utils::linear<num_t>,
                             &class_id,
                             0.5);
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  LOG(INFO) << "Trained with success." << std::endl;  
}
UNIT(NodeLearnNOR) {
  LOG(INFO) << "Train NOR function with Node." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{1.0} },
    { { 0, 1 },{0.0} },
    { { 1, 0 },{0.0} },
    { { 1, 1 },{0.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node<num_t> my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(),
                             utils::linear<num_t>,
                             &class_id,
                             0.5);
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  LOG(INFO) << "Trained with success." << std::endl;  
}

UNIT(NodeLearnNOT) {
  LOG(INFO) << "Train NOT function with Node." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0 },{1.0} },
    { { 1 },{0.0}}
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node<num_t> my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(),
                             utils::linear<num_t>,
                             &class_id,
                             0.5);
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    ASSERT_TRUE(class_id == correct_output);
  }
  LOG(INFO) << "Trained with success." << std::endl;  
}

UNIT(NodeLearnXOR) {
  LOG(INFO) << "Train XOR function with Node." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{0.0} },
    { { 0, 1 },{1.0} },
    { { 1, 0 },{1.0} },
    { { 1, 1 },{0.0} }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }
  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  Node<num_t> my_node(num_features);
  Train(my_node, training_sample_set_with_bias, 0.1, 100);

  for (const auto & training_sample : training_sample_set_with_bias) {
    bool class_id;
    my_node.GetBooleanOutput(training_sample.input_vector(),
                             utils::linear<num_t>,
                             &class_id,
                             0.5);
    bool correct_output = training_sample.output_vector()[0] > 0.5 ? true : false;
    if (class_id != correct_output)
    {
      LOG(WARNING) << "Failed to train. " <<
        " A simple perceptron cannot learn the XOR function." << std::endl;
      //FAIL();
    }
  }
  //LOG(INFO) << "Trained with success." << std::endl;
}


UNIT(NodeRandomiseWeights) {
  std::unique_ptr< Node<num_t> > node_ptr_( new Node<num_t>(
    5, true, 1.
  ) );

  for (auto &w: node_ptr_->m_weights) {
    ASSERT_TRUE(w == 1.);
  }

  node_ptr_->WeightRandomisation(0.1);
  std::cout << "Randomised node weights: ";
  for (auto &w: node_ptr_->m_weights) {
    std::cout << w;
    std::cout << ", ";
  }
  std::cout << std::endl;
}

#if defined(NODETEST_MAIN)

int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  microunit::UnitTester::Run();
  return 0;
}

#endif  // NODETEST_MAIN
