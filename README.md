# Hand-coded Binary Neural Network (BNN) for MNIST

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![NumPy](https://img.shields.io/badge/Core-NumPy-013243?style=flat&logo=numpy)
![PyTorch](https://img.shields.io/badge/DataLoader-PyTorch-EE4C2C?style=flat&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Completed-success)

> A **NumPy-based implementation** of a Binary Neural Network (BNN) trained to classify MNIST digits. This project demonstrates the mathematical mechanics of BNNs, including **1-bit quantization** and the **Straight-Through Estimator (STE)**, built from scratch without high-level deep learning abstraction layers.

## Introduction

Binary Neural Networks (BNNs) drastically reduce memory usage and computational cost by restricting weights and activations to only two values: `+1` and `-1`. This makes them ideal for embedded systems and FPGA hardware implementations where resources are limited.

**Key Highlights of this Project:**
* **From Scratch:** The Forward and Backward propagation logic is implemented entirely using **NumPy** math, demonstrating a deep understanding of neural network calculus.
* **Full Binarization:** Weights, Activations, and Inputs are all binarized to `+1/-1`.
* **Straight-Through Estimator (STE):** Implemented a custom backward pass to handle the non-differentiable nature of the Sign function (addressing the vanishing gradient problem).

## Tech Stack & Architecture

* **Core Logic:** NumPy (Matrix multiplication, Gradients calculation).
* **Data Loading:** PyTorch `torchvision` (Used solely for downloading and transforming the MNIST dataset).
* **Architecture:** Fully Connected (MLP).

### Network Topology
The model consists of 2 Fully Connected layers with Sign Activation functions:

```text
Input (784) --> [Sign] --> FC (512) --> [Sign] --> FC (10) --> Output

Input Layer: 784 neurons (28x28 flattened image), normalized and binarized.

Hidden Layer: 512 neurons, 1-bit Weights.

Output Layer: 10 neurons (Classes 0-9).

Loss Function: MSE Loss (Mean Squared Error).

Performance & Results

Despite the extreme compression (1-bit precision), the model achieves competitive accuracy on the MNIST test set.

Best Test Accuracy:                  91.41%
Training Epochs:                     25
Batch Size:                          64
Optimization,SGD with Learning Rate: 0.001

Compression Efficiency

Theoretical comparison between a standard FP32 model and this BNN:
Standard FP32 Weight: 32 bits.
BNN Weight: 1 bit.
Compression Ratio: ~32x reduction in model size.

Mathematical Implementation (STE & Gradient Flow)
Training BNNs is challenging because the derivative of the Sign function is 0 almost everywhere. This project implements the Straight-Through Estimator (STE) to approximate gradients during backpropagation.

Note on Batch Normalization: 
This implementation intentionally does not use Batch Normalization layers. To compensate for this and ensure gradients can effectively propagate through all layers without vanishing, the STE saturation threshold was increased to 25. This adjustment ensures that the backpropagation process can successfully update weights across the entire depth of the network.

How to Run
1. Clone the repository

git clone [https://github.com/ductri2052005-creator/BNN_Recognition_MNIST.git](https://github.com/ductri2052005-creator/BNN_Recognition_MNIST.git)
cd BNN_Recognition_MNIST

2. Install Dependencies
pip install numpy torch torchvision

3. Run Training
python BNN_code.py

(Note: If you are running on Google Colab, you can open BNN_USING_NUMPY.ipynb directly)

Project Structure
BNN_USING_NUMPY.ipynb: Jupyter Notebook containing the full training pipeline and analysis.
src/: (Optional) Contains modularized python scripts.
bnn_weights_1bit.npz: Exported 1-bit weights after training.

Future Improvements
Implement Convolutional Layers (Binary CNN) using NumPy im2col.
Deploy the trained weights onto an FPGA (Zynq-7000) using Verilog/HLS.
Optimize the threshold value for STE dynamically.