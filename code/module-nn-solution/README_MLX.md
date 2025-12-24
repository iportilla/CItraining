# Notebook Optimization for Apple Silicon (MLX)

This directory contains optimized versions of the neural network notebooks using the Apple MLX framework, designed to run efficiently on Apple Silicon (M1/M2/M3) chips.

## Optimized Notebooks

| Original Notebook | Optimized MLX Version | Description |
|-------------------|-----------------------|-------------|
| `NN-session-1-solution.ipynb` | `NN-session-1-mlx.ipynb` | Binary classification (One-Neuron Network) setup with MLX. Optimized for Apple Silicon. |
| `NN-session-2-solution.ipynb` | `NN-session-2-mlx.ipynb` | Multi-class classification for 18-apps dataset using MLX. Includes custom training loops. |
| `NN-session-3-solution.ipynb` | `NN-session-3-mlx.ipynb` | Advanced hyperparameter tuning and experiments using MLX. Implements manual early stopping and scheduler logic. |

## Key Features

1.  **Apple Silicon Optimization**: Utilizes `mlx.core` and `mlx.nn` for GPU-accelerated training on Mac, significantly faster than CPU-bound Keras on these machines.
2.  **Custom Training Loops**: Implements explicit training loops using `mx.value_and_grad` and `mx.optimize`, offering granular control over the training process.
3.  **Data Compatibility**: Uses the exact same `pandas` and `sklearn` preprocessing pipelines as the original notebooks, ensuring data consistency and comparable results.
4.  **No TensorFlow Dependency**: These notebooks run entirely without TensorFlow/Keras, relying only on the lightweight MLX library.

## Prerequisites

To run these notebooks, you need a Python environment with the following packages installed:

```bash
pip install mlx pandas scikit-learn matplotlib
```

*Note: MLX is only available on macOS with Apple Silicon.*

## How to Use

1.  Open the desired `*-mlx.ipynb` notebook in VS Code or Jupyter Lab.
2.  Select your standard Python kernel (ensure it has `mlx` installed).
3.  Run the cells sequentially.

## Performance & Verification

The notebooks have been verified to produce accuracy results comparable to the original TensorFlow implementations while leveraging the Mac's GPU for training operations. Training loops have been configured for efficiency.
