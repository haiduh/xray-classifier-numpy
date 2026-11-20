# xray-classifier-numpy

A binary chest X-ray classifier (pneumonia vs normal) built from scratch using only NumPy - no PyTorch, no TensorFlow, no ML frameworks.

## Why

Most ML education skips the fundamentals and jumps straight to `model.fit()`. This project exists to understand what actually happens under the hood - forward propagation, backpropagation, and gradient descent implemented from first principles.

## Architecture

- **Input:** 64×64 greyscale chest X-ray, flattened to a 4096-length vector
- **Hidden layer:** 64 neurons, ReLU activation, He initialisation
- **Output layer:** 1 neuron, sigmoid activation
- **Loss:** Binary cross-entropy
- **Optimiser:** Vanilla gradient descent

## Results

| | Accuracy |
|---|---|
| Train (5216 images) | 83.8% |
| Test (624 images) | 67.2% |

The gap between train and test accuracy reflects overfitting - a known limitation of fully-connected networks on raw pixel data. Flattening a 2D image discards all spatial structure, which is what the network needs to generalise well.

## Next Step

Extending this to a CNN to leverage spatial structure via convolutional filters - the standard approach for medical image classification.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kaggle, Paul Mooney.