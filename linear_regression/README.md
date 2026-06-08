# Linear Regression From Scratch

This project implements Single Linear Regression and Multiple Linear Regression from scratch using NumPy without relying on machine learning libraries such as scikit-learn for training.

## Features

* Single Linear Regression
* Multiple Linear Regression
* Gradient Descent Optimization
* R² Score Evaluation
* Loss Tracking
* NumPy Vectorized Operations
* OOP-Based Design

## Mathematical Formulation

### Single Linear Regression

y = mx + b

Where:

* m = slope
* b = intercept

### Multiple Linear Regression

y = Xw + b

Where:

* X = feature matrix
* w = weight vector
* b = intercept

## Optimization

Parameters are learned using Gradient Descent.

For every epoch:

1. Compute predictions
2. Calculate gradients
3. Update weights and intercept
4. Track training loss

## Implemented Methods

### fit(X, y)

Trains the model using Gradient Descent.

### predict(X)

Generates predictions for new data.

### score(X, y)

Computes the R² Score.

### get_params()

Returns learned weights and intercept.


## Comparison With Scikit-Learn

The implementation was validated against scikit-learn's LinearRegression model.

```python
from sklearn.linear_model import LinearRegression
```

Predictions produced by both implementations are nearly identical on non-collinear datasets.

## Learning Outcomes

While building this project, the following concepts were explored:

* Object-Oriented Programming (OOP)
* Gradient Descent
* Matrix Multiplication
* NumPy Vectorization
* Feature Matrices
* Weight Vectors
* R² Score
* Multicollinearity
* Model Evaluation

## Future Improvements

* Feature Scaling
* Mini-Batch Gradient Descent
* Stochastic Gradient Descent
* Regularization (L1/L2)
* Model Serialization
