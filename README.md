# NeuralNetowork
A Python implementation of a fully connected deep neural network trained on the MNIST dataset without using deep learning frameworks.

## Introduction

This project demonstrates the implementation of a deep neural network built entirely from scratch using NumPy. The model is trained on the MNIST handwritten digit dataset to classify digits from 0 to 9. The goal of this project is to understand the internal working of neural networks, including forward propagation, backpropagation, weight initialization, and gradient descent, without relying on high-level deep learning libraries.

## Technologies
Python    
NumPy    
scikit-learn    
OpenML (MNIST dataset)     

## Features

Custom implementation of a deep neural network    
Two hidden layers with sigmoid activation    
Softmax output layer for multi-class classification     
Manual forward propagation and backpropagation   
Stochastic Gradient Descent (SGD) training  
One-hot encoded target labels  
Accuracy evaluation on test data  

## The Process  
Loaded and preprocessed the MNIST dataset.  
Normalized image pixel values to the range [0, 1].  
One-hot encoded the digit labels.  
Initialized network weights using scaled random values.  
Performed forward propagation to generate predictions.  
Calculated errors using backpropagation.  
Updated weights using gradient descent.  
Repeated the process over multiple epochs.  
Evaluated model accuracy after each epoch.  

## What I Learned

How neural networks work internally without using deep learning frameworks  
The importance of data normalization for training stability  
Implementation of forward propagation and backpropagation    
How gradient descent updates weights in a neural network  
Handling multi-class classification using softmax and one-hot encoding  
Debugging shape mismatches and numerical stability issues

## Line-by-Line Code Explanation

#### Lines 1–8: Import required libraries
#### Lines 10–11: Load and normalize the MNIST dataset
#### Line 12–13: Label preprocessing
#### Line 14: Train–test split

## Deep Neural Network Class
#### Line 16: Define the DeepNeuralNetwork class
#### Lines 17–21: Constructor (`__init__`)

* sizes defines the number of neurons in each layer.
* epochs defines how many times the model iterates over the entire dataset.
* l_rate is the learning rate that controls weight updates.
* initialization() is called to initialize network weights.

#### Lines 23–27: Activation functions

* sigmoid is used in hidden layers.
* When derivative=True the derivative of the sigmoid function is returned.
* softmax converts output layer values into class probabilities.

## Weight Initialization
#### Lines 29–33: Define network architecture
* Input layer size is 784 (28×28 images).
* First hidden layer contains 128 neurons.
* Second hidden layer contains 64 neurons.
* Output layer contains 10 neurons (digits 0–9).
  
#### Lines 35–38: Initialize weights

* Weight matrices `W1`, `W2`, and `W3` are created.
* Random initialization is scaled to prevent vanishing or exploding gradients.

## Forward Propagation
#### Line 44–45: Input layer
* Input sample is stored in A0.
* Reshaped into a column vector for matrix multiplication.

#### Lines 47–48: Hidden layer 1

* Z1 is computed using matrix multiplication of W1 and A0.
* A1 is obtained by applying the sigmoid activation function.

#### Lines 50–51: Hidden layer 2

* Z2 is computed using W2 and A1.
* A2 is obtained using the sigmoid function.

#### Lines 53–54: Output layer

* Z2 is computed using W2 and A2.
* A3 is obtained using the softmax function.

#### Line 56: Returns predicted probabilities for each digit class.

## Backpropagation
#### Lines 59–60: Gradient storage

* Network parameters are accessed.
* change_w dictionary stores gradients.

#### Lines 62–63: Output layer error

* Error is computed as the difference between predicted output and true label.
* Gradient for W3 is calculated using A2.

#### Lines 65–66: Hidden layer 2 gradients

* Error is propagated backward using the transpose of W3.
* Sigmoid derivative is applied to Z2.
* Gradient for W2 is calculated.

#### Lines 68–69: Hidden layer 1 gradients

* Error is propagated using the transpose of W2.
* Sigmoid derivative is applied to Z2.
* Gradient for W1 is calculated.

## Training and Evaluation
#### Line 75: Epoch loop
#### Lines 76–79: Stochastic Gradient Descent

* Model processes one training sample at a time.
* Forward propagation generates predictions.
* Backpropagation computes gradients.
* Weights are updated using gradient descent.
#### Line 81: Accuracy computation

* Model predictions are compared with true labels.
* Accuracy is computed on the validation dataset.

#### Lines 82–84: Training progress output

* Displays epoch number, elapsed time, and accuracy.

## Model Execution
* A DeepNeuralNetwork object is created with layer sizes [784, 128, 64, 10].
