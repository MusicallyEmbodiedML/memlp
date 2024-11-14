//============================================================================
// Name : Main.cpp
// Author : David Nogueira
//============================================================================
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdio>

#include "UnitTest.hpp"
#include "MLP.h"


UNIT(MLPLearnAND) {
  LOG(INFO) << "Train AND function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 0.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 150, 0);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnNAND) {
  LOG(INFO) << "Train NAND function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 1.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 0.0 } },
    { { 1, 1 },{ 0.0 } },
    { { 1, 1 },{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 250, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnOR) {
  LOG(INFO) << "Train OR function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 0 },{ 0.0 } },
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 50, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnNOR) {
  LOG(INFO) << "Train NOR function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 1.0 } },
    { { 0, 0 },{ 1.0 } },
    { { 0, 0 },{ 1.0 } },
    { { 0, 1 },{ 0.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 500, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnXOR) {
  LOG(INFO) << "Train XOR function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 1, 500, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnNOT) {
  LOG(INFO) << "Train NOT function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0},{ 1.0 } },
    { { 1},{ 0.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 50, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t> output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnX1) {
  LOG(INFO) << "Train X1 function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 0.0 } },
    { { 1, 0 },{ 1.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });
  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.8, 50, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnX2) {
  LOG(INFO) << "Train X2 function with mlp." << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 1.0 } }
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
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });

  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 50, 0.);

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }
  LOG(INFO) << "Trained with success." << std::endl;
}

UNIT(MLPLearnX2_MiniBatch) {
  LOG(INFO) << "Train X2 function with mini batch mlp." << std::endl;

    // using training_pair_t = std::pair<
    //   std::vector< std::vector<T> >,
    //   std::vector< std::vector<T> >
    // >;

  MLP<num_t>::training_pair_t trainingSet;
  //including bias
  trainingSet.first = {{0,0,1},{0,1,1},{1,0,1},{1,1,1}};
  trainingSet.second = {{0},{1.0},{0},{1.0}};

  size_t num_features = trainingSet.first[0].size();
  size_t num_outputs = trainingSet.second[0].size();
  MLP<num_t> my_mlp({ num_features, 2 ,num_outputs }, { "sigmoid", "linear" });

  //Train MLP
  my_mlp.MiniBatchTrain(trainingSet, 0.5, 50, 2, 0.);

  LOG(INFO) << "Trained with success." << std::endl;
}


#if 1

UNIT(MLPGetWeightsSetWeights) {
  LOG(INFO) << "Train X2 function, read internal weights" << std::endl;

  std::vector<TrainingSample<num_t>> training_set =
  {
    { { 0, 0 },{ 0.0 } },
    { { 0, 1 },{ 1.0 } },
    { { 1, 0 },{ 0.0 } },
    { { 1, 1 },{ 1.0 } }
  };
  bool bias_already_in = false;
  std::vector<TrainingSample<num_t>> training_sample_set_with_bias(training_set);
  //set up bias
  if (!bias_already_in) {
    for (auto & training_sample_with_bias : training_sample_set_with_bias) {
      training_sample_with_bias.AddBiasValue(1);
    }
  }

  LOG(INFO) << "Training set created." << std::endl;

  size_t num_features = training_sample_set_with_bias[0].GetInputVectorSize();
  size_t num_outputs = training_sample_set_with_bias[0].GetOutputVectorSize();
  MLP<num_t> my_mlp({ num_features, 2, num_outputs }, { "sigmoid", "linear" });

  LOG(INFO) << "Training now:" << std::endl;

  //Train MLP
  my_mlp.Train(training_sample_set_with_bias, 0.5, 50, 0.);

  // get layer weights
  std::vector<std::vector<num_t>> weights = my_mlp.GetLayerWeights( 1 );

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      bool predicted_output = output[i] > 0.5 ? true : false;
      std::cout << "PREDICTED OUTPUT IS NOW: " << output[i] << std::endl;
      bool correct_output = training_sample.output_vector()[i] > 0.5 ? true : false;
      ASSERT_TRUE(predicted_output == correct_output);
    }
  }

  // the expected value of the internal weights
  // after training are 1.65693 -0.538749
  // TODO AM fix this test
  //ASSERT_TRUE(  1.6 <= weights[0][0] && weights[0][0] <=  1.7 );
  //ASSERT_TRUE( -0.6 <= weights[0][1] && weights[0][1] <= -0.5 );

  // now, we are going to inject a weight value of 0.0
  // and check that the new output value is nonsense
  std::vector<std::vector<num_t>> zeroWeights = { { 0.0, 0.0 } };

  my_mlp.SetLayerWeights( 1, zeroWeights );

  for (const auto & training_sample : training_sample_set_with_bias) {
    std::vector<num_t>  output;
    my_mlp.GetOutput(training_sample.input_vector(), &output);
    for (size_t i = 0; i < num_outputs; i++) {
      ASSERT_TRUE( -0.0001L <= output[i] && output[i] <= 0.0001L );
    }
  }

  LOG(INFO) << "Trained with success." << std::endl;
}

#endif


#define compare_weights_eq(weights, new_weights)    \
  ASSERT_TRUE(new_weights.size() == weights.size());    \
  for (unsigned int n = 0; n < new_weights.size(); n++) {    \
      ASSERT_TRUE(new_weights[n].size() == weights[n].size());    \
      for (unsigned int k = 0; k < new_weights[n].size(); k++) {    \
          ASSERT_TRUE(new_weights[n][k].size() == weights[n][k].size());    \
          for (unsigned int j = 0; j < new_weights[n][k].size(); j++) {    \
              if (((new_weights[n][k][j] != weights[n][k][j]))) {    \
                  std::printf("new_weights[%d][%d][%d] = %f, weights[same] = %f\n", n, k, j, new_weights[n][k][j], weights[n][k][j]);    \
              }    \
              ASSERT_TRUE(new_weights[n][k][j] == weights[n][k][j]);    \
          }    \
      }    \
  }


#define compare_weights_neq(weights, new_weights)    \
  ASSERT_TRUE(new_weights.size() == weights.size());    \
  for (unsigned int n = 0; n < new_weights.size(); n++) {    \
      ASSERT_TRUE(new_weights[n].size() == weights[n].size());    \
      for (unsigned int k = 0; k < new_weights[n].size(); k++) {    \
          if (((new_weights[n][k].size() != weights[n][k].size()))) {    \
              std::printf("new_weights[%d][%d].size() = %d, weights[same].size() = %d\n", n, k, new_weights[n][k].size(), weights[n][k].size());    \
          }    \
          ASSERT_TRUE(new_weights[n][k].size() == weights[n][k].size());    \
          for (unsigned int j = 0; j < new_weights[n][k].size(); j++) {    \
              if (((new_weights[n][k][j] == weights[n][k][j]))) {    \
                  std::printf("new_weights[%d][%d][%d] = %f, weights[same] = %f\n", n, k, j, new_weights[n][k][j], weights[n][k][j]);    \
              }    \
              ASSERT_TRUE(new_weights[n][k][j] != weights[n][k][j]);    \
          }    \
      }    \
  }


UNIT(MLPGetSetWeightsNew) {
#if 1
    MLP<num_t> mlp(
        {2, 3, 3, 1},
        {"relu", "relu", "sigmoid"}
    );

    MLP<num_t>::mlp_weights new_weights{
        { {1, 2,}, {3, 4,}, {5, 6,}, },
        { {1, 2, 3,}, {4, 5, 6,}, {7, 8, 9}, },
        { {1, 2, 3}, },
    };

    auto weights_before = mlp.GetWeights();
    compare_weights_neq(weights_before, new_weights);
    mlp.SetWeights(new_weights);
    auto weights_after = mlp.GetWeights();

    compare_weights_eq(weights_after, new_weights);
#endif
}


UNIT(MLPSerialise) {
#if 1
    MLP<num_t> mlp(
        {2, 3, 3, 1},
        {"relu", "relu", "sigmoid"}
    );
    auto mlp2 = mlp;

    MLP<num_t>::mlp_weights new_weights{
        { {1, 2,}, {3, 4,}, {5, 6,}, },
        { {1, 2, 3,}, {4, 5, 6,}, {7, 8, 9}, },
        { {1, 2, 3}, },
    };

    mlp.SetWeights(new_weights);
    std::vector<uint8_t> serialised;
    size_t r_head = 0, w_head = 0;

    w_head = mlp.Serialise(w_head, serialised);
    r_head = mlp2.FromSerialised(r_head, serialised);

    MLP<num_t>::mlp_weights mlp_weights = mlp.GetWeights();
    MLP<num_t>::mlp_weights mlp_2_weights = mlp2.GetWeights();

    ASSERT_TRUE(mlp_weights == new_weights);
    ASSERT_TRUE(mlp_weights == mlp_2_weights);
#endif
}


UNIT(MLPUpdateWeigthsMSE) {
    // Weights from PyTorch reference
    // See: https://colab.research.google.com/drive/1DgNlTbtpo8XD8YEmGqpKu2ma4BiHRW_B
    std::vector< std::vector< std::vector<num_t> > > weights {
        { { -0.5226030349731445, 0.09498977661132812 }
        , { -0.28878581523895264, -0.34213435649871826 }
        , { -0.647223949432373, -0.10152232646942139 }
        , { 0.42996883392333984, -0.019827842712402344 }
        },
        { { 0.4287152886390686, 0.4463331699371338, -0.3849239945411682, 0.1832484006881714 }
        , { 0.1930738091468811, 0.19118380546569824, -0.06816577911376953, 0.125044047832489 }
        , { 0.12581825256347656, 0.48664939403533936, -0.4792138934135437, -0.18973785638809204 }
        , { -0.33640098571777344, -0.24253511428833008, 0.09816908836364746, 0.3049418330192566 }
        },
        { { -0.3781070113182068, 0.32520592212677, 0.41765105724334717, -0.24645352363586426 }
        }
    };
    // In and out
    const nd_vector input { {1., /*plus bias*/ 1.} };
    const nd_vector expected_output { {1.} };
    // Weights after update
    std::vector< std::vector< std::vector<num_t> > > expected_weights {
        { { -0.521845817565918, 0.0957469642162323 }
        , { -0.2854526937007904, -0.33880123496055603 }
        , { -0.6493763327598572, -0.10367471724748611 }
        , { 0.04061141610145569, -0.4091852605342865 }
        },
        { { 0.43217340111732483, 0.4514354467391968, -0.37886887788772583, -0.14843356609344482 }
        , { 0.1900908648967743, 0.1867826282978058, -0.07338888943195343, 0.41115042567253113 }
        , { 0.1220010370016098, 0.4810172915458679, -0.48589780926704407, 0.17638686299324036 }
        , { -0.3341711163520813, -0.23924507200717926, 0.1020735502243042, 0.09106685221195221 }
        },
        { { -0.22057375311851501, 0.4321286976337433, 0.2506052851676941, 0.02582395076751709 }
        }
    };

    // Actual unit test
    std::vector<size_t> nodes{ 2, 4, 4, 1 };
    std::vector<std::string> activations{ "relu", "tanh", "linear" };
    auto model = std::make_unique< MLP<num_t> >(
        nodes, activations
    );
    // Set weights
    for (unsigned int n=0; n<weights.size(); n++) {
        model->SetLayerWeights(n, weights[n]);
    }
    // Train once
    model->Train(
        std::make_pair(input, expected_output),
        1.,    // learning_rate
        1,     // max_iteration
        0.,    // min_error_cost
        false  // output_log
    );

    // Check weights
    for (unsigned int l=0; l<model->GetNumLayers(); l++) {
        auto layer_weights = model->GetLayerWeights(l);
        for (unsigned int n=0; n<layer_weights.size(); n++) {
            for (unsigned int k=0; k<layer_weights[n].size(); k++) {

                ASSERT_TRUE(utils::is_close(
                    layer_weights[n][k],
                    expected_weights[l][n][k]
                ));

            }  // for k
        }  // for n
    }  // for l
}


UNIT(MLPDrawWeights) {
  std::unique_ptr< MLP<num_t> > mlp_ptr (new MLP<num_t> (
    {3, 3, 1},
    {"relu", "sigmoid"}
  ));
#if 1
  MLP<num_t>::mlp_weights weights_1(mlp_ptr->GetWeights());

  LOG(INFO) << "About to draw weights..." << std::endl;
  mlp_ptr->DrawWeights();
  LOG(INFO) << "Weights drawn." << std::endl;

  MLP<num_t>::mlp_weights weights_2(mlp_ptr->GetWeights());

  compare_weights_neq(weights_1, weights_2);

  LOG(INFO) << "About to draw weights..." << std::endl;
  mlp_ptr->DrawWeights();
  LOG(INFO) << "Weights drawn." << std::endl;

  MLP<num_t>::mlp_weights weights_3(mlp_ptr->GetWeights());

  compare_weights_neq(weights_1, weights_3);
  compare_weights_neq(weights_2, weights_3);
#endif
}

UNIT(MLPMoveWeights) {
  std::unique_ptr< MLP<num_t> > mlp_ptr (new MLP<num_t> (
    {3, 3, 1},
    {"relu", "sigmoid"}
  ));

  MLP<num_t>::mlp_weights weights_1(mlp_ptr->GetWeights());

  LOG(INFO) << "About to move weights..." << std::endl;
  mlp_ptr->MoveWeights(0.1);
  LOG(INFO) << "Weights moved." << std::endl;

  MLP<num_t>::mlp_weights weights_2(mlp_ptr->GetWeights());

  compare_weights_neq(weights_1, weights_2);

  LOG(INFO) << "About to move weights..." << std::endl;
  mlp_ptr->MoveWeights(0.1);
  LOG(INFO) << "Weights moved." << std::endl;

  MLP<num_t>::mlp_weights weights_3(mlp_ptr->GetWeights());

  compare_weights_neq(weights_1, weights_3);
  compare_weights_neq(weights_2, weights_3);
}


#if defined(MLPTEST_MAIN)

int main(int argc, char* argv[]) {
  START_EASYLOGGINGPP(argc, argv);
  microunit::UnitTester::Run();
  return 0;
}

#endif  // MLPTEST_MAIN