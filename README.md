# Twitter-sentimental-classification

This repository provides four different methods using scikit-learn to classify tweets based on their sentiments. The first step involves data mining to extract useful information from the training tweets.

# Setup
Python 3.x

Torch

sklearn

pandas

numpy

nltk

gc


# Algorithm
## 1. Logistic Regression
### Architechture:
4 Linear layers

3 ReLu activation
### Optimizer:
Adam optimizer
### Accuracy: 0.9923

## 2. Regularized Logistic Regression
### Architechture:
4 Linear layers

3 ReLu activation
### Optimizer:
Adam optimizer with weight decay
### Accuracy: 0.9888

## 3. Decision Tree
### Architechture:
A decision tree with depth = 20
### Accuracy: 0.9623

## 4. Decision Forest
### Architechture:
A decision tree with depth = 40
### Accuracy: 0.9853

