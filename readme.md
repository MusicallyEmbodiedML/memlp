# MEMLP Library

## Introduction

The MEMLP (Microcontroller Embedded Multi-Layer Perceptron) library is a lightweight and efficient library designed for implementing machine learning models on resource-constrained devices such as microcontrollers. It supports the Arduino framework and microcontrollers like the Raspberry Pi Pico. 

### Key Features:
- **Platform Compatibility**: Works seamlessly with C++11 and is optimized for microcontrollers.
- **Reinforcement Learning**: Includes support for reinforcement learning algorithms.
- **Customizability**: Allows users to define custom architectures, activation functions, and loss functions.
- **Replay Memory**: Implements replay memory for reinforcement learning tasks.

### Supported Platforms:
- Arduino IDE
- Raspberry Pi Pico
- Other microcontrollers with C++11 support

## Installation Instructions

### Prerequisites:
- **Arduino IDE**: Ensure you have the latest version installed.
- **Earle Philhower's Arduino-Pico Library**: Follow the [installation instructions](https://github.com/earlephilhower/arduino-pico) on the project's GitHub page.

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/memlp.git
   ```
2. Create a folder named `src` in your project directory.
3. Add the MEMLP library as a submodule:
   ```bash
   git submodule add https://github.com/your-repo/memlp.git src/memlp
   ```
4. Include the library in your project:
   ```cpp
   #include "src/memlp/MLP.h"
   ```

## Usage Examples

### Initializing the Library
```cpp
#include "src/memlp/MLP.h"

MLP<float> my_mlp({6, 16, 8, 8, 12}, 
                  {ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR, ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::SIGMOID});
```

### Training a Model
```cpp
std::vector<std::vector<float>> features = {{0.1, 0.2}, {0.3, 0.4}};
std::vector<std::vector<float>> labels = {{1.0}, {0.0}};
my_mlp.Train({features, labels}, 0.01, 1000, 0.001);
```

### Running Inference
```cpp
std::vector<float> input = {0.1, 0.2};
std::vector<float> output;
my_mlp.GetOutput(input, &output);
```

### Implementing Reinforcement Learning
```cpp
my_mlp.SmoothUpdateWeights(target_mlp, 0.1f);
```

## API Documentation

### Classes and Methods

#### `MLP`
- **Constructor**:
  ```cpp
  MLP(const std::vector<size_t> &layers_nodes, const std::vector<ACTIVATION_FUNCTIONS> &layers_activfuncs, loss::LOSS_FUNCTIONS loss_function = loss::LOSS_FUNCTIONS::LOSS_MSE);
  ```
  - **Parameters**:
    - `layers_nodes`: Number of nodes in each layer.
    - `layers_activfuncs`: Activation functions for each layer.
    - `loss_function`: Loss function to use (default: Mean Squared Error).

- **Methods**:
  - `void Train(const training_pair_t& training_sample_set_with_bias, float learning_rate, int max_iterations, float min_error_cost, bool output_log);`
    - Trains the model using the provided dataset.
  - `void GetOutput(const std::vector<T> &input, std::vector<T> *output);`
    - Runs inference on the input data.
  - `void SmoothUpdateWeights(std::shared_ptr<MLP<T>> anotherMLP, const float alpha);`
    - Updates weights for reinforcement learning.

#### `Dataset`
- **Methods**:
  - `bool Add(const std::vector<float> &feature, const std::vector<float> &label);`
    - Adds a new data point to the dataset.
  - `std::pair<DatasetVector, DatasetVector> Sample(bool with_bias = true);`
    - Samples data from the dataset.

#### `ReplayMemory`
- **Methods**:
  - `void add(trainingItem &tp, size_t timestamp);`
    - Adds a training item to the replay memory.
  - `std::vector<trainingItem> sample(size_t nMemories);`
    - Samples a batch of training items.

### Supported Algorithms
- **Activation Functions**:
  - Sigmoid
  - Tanh
  - ReLU
  - Linear
- **Loss Functions**:
  - Mean Squared Error (MSE)

## Contributing Guidelines

We welcome contributions to the MEMLP library! To contribute:
1. Fork the repository and create a new branch.
2. Follow the coding standards outlined in the `CONTRIBUTING.md` file.
3. Submit a pull request with a detailed description of your changes.

### Setting Up a Development Environment
- Install the required tools (e.g., C++ compiler, Arduino IDE).
- Clone the repository and ensure all dependencies are installed.

## License

This library is distributed under the Mozilla Public License (MPL). See the `LICENSE` file for more details.

## Contact and Support

For support or to report issues, please use:
- **GitHub Issues**: [https://github.com/your-repo/memlp/issues](https://github.com/your-repo/memlp/issues)
- **Email**: support@yourdomain.com
