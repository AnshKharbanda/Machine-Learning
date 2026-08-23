# KNN Classifier From Scratch

## 1. Intuition

**K-Nearest Neighbors (KNN)** is a supervised, non-parametric, lazy learning algorithm.

For a new data point:

1. Calculate its distance from training points.
2. Find the K nearest points.
3. Take their labels.
4. Predict the class with the most votes.

**Core idea:** similar points tend to have similar labels.

## 2. Bias, Variance, Overfitting & Underfitting

K controls the bias-variance tradeoff.

### Small K

* Low bias
* High variance
* Sensitive to noise and outliers
* Can **overfit**

Example: K = 1 can simply memorize the nearest training point.

### Large K

* High bias
* Low variance
* Smoother decision boundary
* Can **underfit**

Therefore, K should usually be selected using validation or cross-validation.

## 3. Pros

* Simple and intuitive
* Easy to implement
* No assumptions about the shape of the decision boundary
* Can handle nonlinear decision boundaries
* Naturally supports multiclass classification
* Useful when local similarity is meaningful

## 4. Cons

* Prediction can be slow for large datasets
* Stores the training data, so memory usage can be high
* Sensitive to feature scaling
* Sensitive to irrelevant features and noise
* Performs poorly in very high-dimensional spaces due to the curse of dimensionality
* Choosing K affects performance significantly

## 5. Requirements Before Using KNN

### Feature scaling

KNN is distance-based, so features should generally be on comparable scales.

Common choices:

* Standardization
* Min-Max scaling

Example problem:

```text
Age:     18–60
Income:  20,000–200,000
```

Income can dominate the distance if features are not scaled.

### Other considerations

* Remove or handle missing values.
* Encode categorical features appropriately.
* Remove or reduce irrelevant features.
* Make sure the chosen distance metric represents similarity meaningfully.
* For high-dimensional data, consider dimensionality reduction such as PCA.

## 6. When to Use KNN

Use KNN when:

* The dataset is small or medium-sized.
* Similarity between nearby points is meaningful.
* The relationship is nonlinear.
* You want a simple baseline model.
* Interpretability through nearest examples is useful.

## 7. When Not to Use KNN

Avoid or reconsider KNN when:

* The dataset is extremely large.
* Predictions need to be extremely fast.
* The feature space is very high-dimensional.
* Distance between samples is not meaningful.
* There are many irrelevant features.
* The data requires a model that learns a compact representation instead of storing training examples.

## 8. My Final KNN Implementation

The current implementation contains:

* `fit()`
* `predict()`
* `accuracy()`
* NumPy arrays
* Input validation for K
* X/Y sample-count validation
* Vectorized squared Euclidean distance
* `np.argsort()` for nearest-neighbor selection
* Dictionary-based majority voting
* Support for multiple test samples and arbitrary class labels

### Prediction pipeline

```text
Test point
    ↓
Squared Euclidean distances
    ↓
np.argsort()
    ↓
K nearest indices
    ↓
Class-frequency dictionary
    ↓
Majority vote
    ↓
Prediction
```

Accuracy is calculated as:

```text
correct predictions / total predictions
```

## 9. Future Improvements

The current implementation is intentionally a basic KNN classifier.

Possible future improvements:

* Add confusion matrix, precision, recall and F1-score.
* Support distance-weighted voting.
* Allow different distance metrics such as Manhattan distance.
* Improve input/shape validation.
* Explore efficient nearest-neighbor structures such as KD-trees for larger datasets.


