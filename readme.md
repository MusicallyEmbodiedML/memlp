# MEMLP Library

## Using with Arduino IDE

Create a folder called `src` then add this library as a submodule.

Example:

```
#include "src/memlp/MLP.h"

MLP<float> my_mlp({ 6, 16, 8, 8, 12 }, { ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::LINEAR, ACTIVATION_FUNCTIONS::RELU, ACTIVATION_FUNCTIONS::SIGMOID });
```

