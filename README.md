# Grokking Deep Learning: Code Reproduction & Theoretical Deep-Dive

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Tugas 2 (Enrichment for Deep Learning Classes) — Individual Task**  
> **Student Name:** Yoga Helmi Ilham Saputra  
> **GitHub Profile:** [@yogahelmi](https://github.com/yogahelmi)  
> **Repository:** [yogahelmi/Grokking-Deep-Learning](https://github.com/yogahelmi/Grokking-Deep-Learning)

---

## 1. Project Objective & Academic Context
This repository contains the complete code reproduction, mathematical derivations, and structured theoretical deep-dives for Andrew W. Trask's seminal book **"Grokking Deep Learning"** (Manning Publications), cross-referenced with geometric and algebraic insights from **"Practical Linear Algebra for Data Science"** by Mike X Cohen (O'Reilly Media).

The primary objective is to cultivate an intuitive, bottom-up mechanical understanding of artificial neural networks without relying on high-level black-box frameworks (such as PyTorch or TensorFlow). Every fundamental concept—from scalar prediction and analytical gradient descent to multidimensional backpropagation and 2D convolution—is built from the ground up using **pure Python and NumPy**.

---

## 2. Repository Roadmap & Google Colab Links

| Chapter | Topic / Milestone | Key Theoretical Concepts | Interactive Notebook |
| :---: | :--- | :--- | :---: |
| **01** | **Introducing Deep Learning** | Parametric vs. Non-Parametric modeling, AI/ML/DL hierarchy, automation of intelligence. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/01.introduction.ipynb) |
| **02** | **Fundamental Concepts: How Machines Learn** | Universal Trilogy (*Predict, Compare, Learn*), Vector spaces $\mathbb{R}^d$, Dot Product geometry ($\mathbf{a}^T\mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/02.fundamental-concepts.ipynb) |
| **03** | **Forward Propagation: Making Predictions** | Scalar/Vector/Matrix forward maps ($\mathbf{\hat{y}} = \mathbf{W}\mathbf{x}$), feature filtering, stacked linear layers. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/03.making-predictions.ipynb) |
| **04** | **Gradient Descent: Neural Learning** | Squared Error ($E = (\hat{y}-y)^2$), Hot/Cold vs. Calculus, derivative derivation ($\frac{dE}{dw}=2\delta x$), divergence threshold ($\alpha > \frac{1}{x^2}$). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/04.gradient-descent.ipynb) |
| **05** | **Generalizing Gradient Descent** | Multi-input/multi-output updates, credit assignment, outer product rank-1 updates ($\nabla_\mathbf{W} E = \mathbf{\delta} \mathbf{x}^T$). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/05.generalizing-gradient-descent.ipynb) |
| **06** | **Intro to Backpropagation** | The Streetlights / XOR dilemma, Proof of Linearity Collapse ($\mathbf{W}_2\mathbf{W}_1\mathbf{x} = \mathbf{W}_{net}\mathbf{x}$), ReLU activation, Adjoint transpose ($\mathbf{W}^T$). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/06.intro-to-backpropagation.ipynb) |
| **07** | **How to Picture Neural Networks** | 10,000-ft view, Correlation Creation vs. Detection, Cover's Theorem, Space Warping and Manifold Bending. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/07.picturing-neural-networks.ipynb) |
| **08** | **Regularization & Batching** | MNIST digit benchmark, Overfitting vs. Signal, Inverted Dropout ($2\times$ scaling), BLAS Mini-batch GEMM speedup. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/08.regularization-and-batching.ipynb) |
| **09** | **Activation Functions & Probabilities** | Zero-centered $\tanh(x)$, Softmax probability simplex projection, Cross-Entropy loss ($\nabla_\mathbf{z} L = \mathbf{\hat{y}}-\mathbf{y}$), Vanishing gradients. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/09.activation-functions.ipynb) |
| **10** | **Convolutional Neural Networks (CNNs)** | 2D Spatial topology, Local receptive fields, Weight sharing, Block Toeplitz linear operators, 16-kernel edge detectors. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yogahelmi/Grokking-Deep-Learning/blob/main/10.convolutional-neural-networks.ipynb) |

---

## 3. General Chapter Summaries

### Chapter 01: Introducing Deep Learning
Introduces the historical and conceptual transition from manual feature engineering to representation learning. It formalizes parametric models (fixed-parameter functions with $O(1)$ inference) versus non-parametric models (instance-storing lookup methods), demonstrating how linear algebra provides a universal mathematical medium for modeling intelligence.

### Chapter 02: Fundamental Concepts — How Do Machines Learn?
Deconstructs the learning process into the algorithmic triad: **Predict, Compare, Learn**. Using *Practical Linear Algebra* as reference, vectors, matrices, and tensors are analyzed geometrically. The vector dot product $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$ is established as the fundamental correlation detector across all neural architectures.

### Chapter 03: Forward Propagation — Intro to Neural Prediction
Walks through the mechanics of inference across five architectural configurations: single-input single-output, multi-input single-output, single-input multi-output, multi-input multi-output, and multi-layer perceptrons. It proves that forward propagation is fundamentally chained matrix-vector multiplication $\mathbf{y} = \mathbf{W}_2 \mathbf{W}_1 \mathbf{x}$.

### Chapter 04: Gradient Descent — Intro to Neural Learning
Investigates error evaluation and parameter optimization. Contrasts heuristic "Hot and Cold" perturbation searches with analytical calculus. Derives the instantaneous gradient $\frac{\partial E}{\partial w} = 2(\hat{y} - y)x$ and proves mathematically why excessive learning rates ($\alpha > 1/x^2$) lead to unstable overshooting and catastrophic divergence.

### Chapter 05: Generalizing Gradient Descent — Learning Multiple Weights at a Time
Extends gradient optimization to multi-dimensional weight vectors and full matrices. Analyzes the credit assignment problem through parameter freezing experiments. Proves that the gradient of an entire weight matrix is computed via the vector outer product $\nabla_\mathbf{W} E = \mathbf{\delta} \mathbf{x}^T$, producing rank-1 update matrices that accumulate into full-rank transformations.

### Chapter 06: Intro to Backpropagation — Building Your First Deep Neural Network
Addresses the failure of linear networks on the Streetlights / XOR problem. Proves that stacking purely linear layers collapses mathematically into a single linear map. Introduces the Rectified Linear Unit (ReLU) to break linearity and derives the multivariate backpropagation algorithm, revealing the transpose matrix $\mathbf{W}^T$ as the adjoint operator projecting error deltas backward.

### Chapter 07: How to Picture Neural Networks — In Your Head and On Paper
Provides the high-level geometric mental model of deep learning. Demonstrates how hidden layers synthesize new intermediate correlations when raw input features have zero correlation with targets. Links this representational power to Cover's Theorem and visualizes how neural layers warp high-dimensional manifolds to render tangled classes linearly separable.

### Chapter 08: Intro to Regularization — Learning Signal and Ignoring Noise
Tackles generalization and overfitting on the MNIST handwritten digits benchmark. Implements early stopping, inverted dropout ($p=0.5$ with $2\times$ activation compensation), and mini-batch gradient descent. Explains the computational advantages of BLAS matrix-matrix multiplication (GEMM) over sequential sample loops.

### Chapter 09: Intro to Activation Functions — Modeling Probabilities
Upgrades the MNIST network to model valid probability distributions over multi-class outputs. Implements zero-centered hyperbolic tangent ($\tanh$) for hidden units and numerically stable Softmax for output units. Analyzes categorical cross-entropy loss, proving that its gradient with respect to pre-activation logits simplifies to prediction minus target ($\mathbf{\hat{y}} - \mathbf{y}$).

### Chapter 10: Intro to Convolutional Neural Networks — Learning Edges and Corners
Resolves the limitations of dense layers on visual data (parameter explosion and lack of translation invariance). Implements a complete 2D convolutional network from scratch in NumPy, sliding 16 $3 \times 3$ filter kernels across $28 \times 28$ digit images, sharing weights across all spatial coordinates, and backpropagating errors directly into the convolutional kernels.

---

## 4. The Linear Algebra Bridge: Theoretical Foundations
This repository explicitly integrates concepts from **"Practical Linear Algebra for Data Science"** (Mike X Cohen):

```
+------------------------------------+---------------------------------------------------------+
| Linear Algebra Concept             | Neural Network Manifestation                            |
+------------------------------------+---------------------------------------------------------+
| Vector (x in R^d)                  | Input feature states and neural layer activations       |
| Dot Product (a^T b)                | Feature detector / correlation filter (w^T x)           |
| Linear Transformation (Ax)         | Layer forward propagation (W x + b)                     |
| Matrix Composition (B A x)         | Multi-layer linear networks (collapses to single map)   |
| Non-Linear Partitioning            | ReLU piecewise-linear coordinate folding                |
| Gradient Vector (grad_w E)         | Vector of steepest ascent in parameter space            |
| Outer Product (u v^T)              | Rank-1 weight matrix update (delta x^T)                 |
| Matrix Transpose (A^T)             | Adjoint backward error projection in backpropagation    |
| Block Toeplitz Operator            | 2D discrete convolution with shared filter kernels      |
+------------------------------------+---------------------------------------------------------+
```

---

## 5. Getting Started & Local Installation

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- Git

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/yogahelmi/Grokking-Deep-Learning.git
cd Grokking-Deep-Learning
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:
```bash
jupyter notebook
```

---

## 6. Academic Integrity & Acknowledgements
- Code implementations reproduce and build upon the educational designs in **"Grokking Deep Learning"** by Andrew W. Trask (Manning Publications, 2019).
- Linear algebra frameworks are referenced from **"Practical Linear Algebra for Data Science"** by Mike X Cohen (O'Reilly Media, 2022).
- All explanations, LaTeX derivations, visualizations, and comparative benchmarks were created as part of **Enrichment for Deep Learning Classes (Tugas 2)**.
