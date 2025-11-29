# dpl

A deep learning framework based on O'Reilly's "Deep Learning from Scratch".

## Features

- **JAX Metal support**: Optimized for macOS with Apple Silicon
- Automatic differentiation for gradient computation
- Implementations of MLP, CNN, RNN/LSTM models
- Optimizers: SGD, Adam

## Installation

```bash
uv sync
```

## Usage

```python
import dpl
import dpl.functions as F
from dpl.models import MLP

# Define model
model = MLP((10, 5))

# Training
x = dpl.Variable(...)
y = model(x)
loss = F.mean_squared_error(y, target)
```

See `examples/` for more details.

## Requirements

- Python >=3.13
- JAX 0.5.0
- JAX Metal 0.1.1 (Apple Silicon)
