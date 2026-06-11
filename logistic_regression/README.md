# Logistic Regression From Scratch

A NumPy-based implementation of Logistic Regression built completely from scratch without using machine learning libraries such as scikit-learn for training.

## Features

* Logistic Regression implementation from scratch
* Gradient Descent optimization
* Sigmoid activation function
* Binary Cross-Entropy Loss
* Prediction and Accuracy Score methods
* Vectorized NumPy operations
* Comparison with scikit-learn implementation

## Mathematical Foundation

### Hypothesis Function

The model computes:

[
z = XW + b
]

and applies the sigmoid activation:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

to obtain probabilities between 0 and 1.

### Binary Cross-Entropy Loss

[
L=-\frac{1}{m}\sum_{i=1}^{m}
\left[
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
\right]
]

where:

* (m) = number of training samples
* (y) = actual label
* (\hat y) = predicted probability

### Gradient Computation

The gradients used for optimization are:

[
dw=\frac{1}{m}X^T(\hat y-y)
]

[
db=\frac{1}{m}\sum(\hat y-y)
]

### Parameter Update

Weights and bias are updated using Gradient Descent:

[
W = W - \alpha dw
]

[
b = b - \alpha db
]

where (\alpha) is the learning rate.

## Project Structure

```text
logistic_regression/
│
├── lr_1.py
├── test.py
└── README.md
```

## Results

The implementation was validated against scikit-learn's Logistic Regression model and achieved comparable accuracy on test datasets.

## Learning Outcomes

This project demonstrates understanding of:

* Binary Classification
* Sigmoid Function
* Cross-Entropy Loss
* Gradient Descent
* Vectorized Linear Algebra
* Model Evaluation
* Machine Learning Fundamentals

## Future Improvements

* L1/L2 Regularization
* Mini-Batch Gradient Descent
* Early Stopping
* Loss Tracking and Visualization
* Multiclass Logistic Regression (Softmax)
* Probability Prediction Method
