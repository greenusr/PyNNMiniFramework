# Neural Network Framework (Python – Fully Object-Oriented, From Scratch)

## Overview
This project is a **fully object-oriented neural network framework** implemented entirely in **pure Python**, without any external libraries (not even NumPy).  
It demonstrates the core principles of neural networks, including:

- Forward propagation  
- Backpropagation  
- Gradient descent  
- Activation functions  
- Loss functions  

The project also includes a **practical example**: training a neural network to classify the **Iris flower dataset**, where you can observe **loss convergence** and **parameter updates** during training.

---

## Features

- **Fully Object-Oriented Design**  
  Core components such as layers, nodes, activation functions, and loss functions are separated into interfaces and implementations.

- **Neural Network Components**  
  - Forward propagation and backpropagation implemented from scratch  
  - Gradient descent optimization  
  - Supports common activation functions: Sigmoid, Softmax, Linear  
  - Supports basic loss functions: Mean Squared Error (MSE), Cross-Entropy  

- **Training Example**  
  - Train a neural network to classify Iris flowers  
  - Includes configurable **epochs** and **learning rate**  
  - Real-time observation of **parameter updates** (weights and biases)  

---

## Usage

1. **Clone the repository:**

```bash
git clone <repository_url>
```

2.	Run the training script:

```bash
python3 main.py
```
3.	Observe the output:
- Training loss updates per epoch
- Parameter changes (weights and biases)
- Final classification results on the Iris dataset

---
## Requirements
- Python 3.x
- No external libraries required (pure Python)

---

## Learning Outcomes

By completing this project, you will:
- Understand **how neural networks work under the hood**
- Learn **forward/backward propagation and gradient descent**
- Practice **object-oriented programming in Python**
- Gain hands-on experience with **a simple ML classification problem**

---

## Future Improvements
- Add optimization algorithms like **Adam**, **RMSProp**
- Extend support for **convolutional** and **recurrent layers**
- Improve **data handling** for larger datasets

---

